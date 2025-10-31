#include "criu-log.h"
#include "plugin.h"
#include "util.h"
#include "cr_options.h"
#include "pid.h"
#include "proc_parse.h"
#include "seize.h"
#include "fault-injection.h"
#include "pstree.h"

#include <common/list.h>
#include <compel/infect.h>

#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <time.h>
#include <pthread.h>
#include <limits.h>

/* cuda-checkpoint binary should live in your PATH */
#define CUDA_CHECKPOINT "cuda-checkpoint"

/* cuda-checkpoint --action flags */
#define ACTION_LOCK	  "lock"
#define ACTION_CHECKPOINT "checkpoint"
#define ACTION_RESTORE	  "restore"
#define ACTION_UNLOCK	  "unlock"

typedef enum {
	CUDA_TASK_RUNNING = 0,
	CUDA_TASK_LOCKED,
	CUDA_TASK_CHECKPOINTED,
	CUDA_TASK_UNKNOWN = -1
} cuda_task_state_t;

#define CUDA_CKPT_BUF_SIZE (128)

#ifdef LOG_PREFIX
#undef LOG_PREFIX
#endif
#define LOG_PREFIX "cuda_plugin: "

/* Disable plugin functionality if cuda-checkpoint is not in $PATH or driver
 * version doesn't support --action flag
 */
bool plugin_disabled = false;

bool plugin_added_to_inventory = false;

/* Enable/disable async GPU restore overlap (for benchmarking) */
static bool async_restore_enabled = true; /* Set to true for overlap measurements */

/* Timing/profiling infrastructure */
struct timing_stats {
	unsigned long pause_devices_us;
	unsigned long checkpoint_devices_us;
	unsigned long resume_devices_us;
	unsigned long total_cuda_checkpoint_calls;
	unsigned long total_cuda_checkpoint_us;
};

static struct timing_stats g_timing_stats = { 0 };

static inline unsigned long timespec_diff_us(struct timespec *start, struct timespec *end)
{
	unsigned long diff_sec = end->tv_sec - start->tv_sec;
	long diff_nsec = end->tv_nsec - start->tv_nsec;
	return (diff_sec * 1000000UL) + (diff_nsec / 1000);
}

#define TIMING_START(var) struct timespec var; clock_gettime(CLOCK_MONOTONIC, &var)
#define TIMING_END(var, accum)                                                                                         \
	do {                                                                                                           \
		struct timespec __end;                                                                                 \
		clock_gettime(CLOCK_MONOTONIC, &__end);                                                               \
		accum += timespec_diff_us(&var, &__end);                                                              \
	} while (0)

struct pid_info {
	int pid;
	char checkpointed;
	cuda_task_state_t initial_task_state;
	int restore_tid; /* Cached restore TID to avoid redundant lookups */
	struct list_head list;
};

/* Used to track which PID's we've paused CUDA operations on so far so we can
 * release them after we're done with the DUMP
 */
static LIST_HEAD(cuda_pids);

/* Async restore support */
struct async_restore_task {
	pthread_t thread;
	int pid;
	int result;
	bool started;
	bool completed;
	struct list_head list;
};

static LIST_HEAD(async_restore_tasks);
static pthread_mutex_t async_restore_lock = PTHREAD_MUTEX_INITIALIZER;
static bool async_restore_triggered = false; /* Track if we've started async restore */

static void dealloc_pid_buffer(struct list_head *pid_buf)
{
	struct pid_info *info;
	struct pid_info *n;

	list_for_each_entry_safe(info, n, pid_buf, list) {
		list_del(&info->list);
		xfree(info);
	}
}

static void dealloc_async_restore_tasks(void)
{
	struct async_restore_task *task;
	struct async_restore_task *n;

	pthread_mutex_lock(&async_restore_lock);
	list_for_each_entry_safe(task, n, &async_restore_tasks, list) {
		list_del(&task->list);
		xfree(task);
	}
	pthread_mutex_unlock(&async_restore_lock);
}

static int add_pid_to_buf(struct list_head *pid_buf, int pid, cuda_task_state_t state, int restore_tid)
{
	struct pid_info *new = xmalloc(sizeof(*new));

	if (new == NULL) {
		return -1;
	}

	new->pid = pid;
	new->checkpointed = 0;
	new->initial_task_state = state;
	new->restore_tid = restore_tid; /* Cache the restore TID */
	list_add_tail(&new->list, pid_buf);

	pr_debug("Cached restore_tid %d for pid %d (state: %d)\n", restore_tid, pid, state);

	return 0;
}

static int launch_cuda_checkpoint(const char **args, char *buf, int buf_size)
{
#define READ  0
#define WRITE 1
	int fd[2], buf_off;
	TIMING_START(launch_start);

	if (pipe(fd) != 0) {
		pr_perror("Couldn't create pipes for reading cuda-checkpoint output");
		return -1;
	}

	buf[0] = '\0';

	int child_pid = fork();
	if (child_pid == -1) {
		pr_perror("Failed to fork to exec cuda-checkpoint");
		close(fd[READ]);
		close(fd[WRITE]);
		return -1;
	}

	if (child_pid == 0) { // child
		if (dup2(fd[WRITE], STDOUT_FILENO) == -1) {
			pr_perror("unable to clone fd %d->%d", fd[WRITE], STDOUT_FILENO);
			_exit(EXIT_FAILURE);
		}
		if (dup2(fd[WRITE], STDERR_FILENO) == -1) {
			pr_perror("unable to clone fd %d->%d", fd[WRITE], STDERR_FILENO);
			_exit(EXIT_FAILURE);
		}
		close(fd[READ]);

		close_fds(STDERR_FILENO + 1);

		execvp(args[0], (char **)args);

		/* We can't use pr_error() as log file fd is closed. */
		fprintf(stderr, "execvp(\"%s\") failed: %s\n", args[0], strerror(errno));

		_exit(EXIT_FAILURE);
	}

	close(fd[WRITE]);
	buf_off = 0;
	/* Reserve one byte for the null charracter. */
	buf_size--;
	while (buf_off < buf_size) {
		int bytes_read;
		bytes_read = read(fd[READ], buf + buf_off, buf_size - buf_off);
		if (bytes_read == -1) {
			pr_perror("Unable to read output of cuda-checkpoint");
			goto err;
		}
		if (bytes_read == 0)
			break;
		buf_off += bytes_read;
	}
	buf[buf_off] = '\0';

	/* Clear out any of the remaining output in the pipe in case the buffer wasn't large enough */
	while (true) {
		char scratch[1024];
		int bytes_read;
		bytes_read = read(fd[READ], scratch, sizeof(scratch));
		if (bytes_read == -1) {
			pr_perror("Unable to read output of cuda-checkpoint");
			goto err;
		}
		if (bytes_read == 0)
			break;
	}
	close(fd[READ]);

	int status, exit_code = -1;
	if (waitpid(child_pid, &status, 0) == -1) {
		pr_perror("Unable to wait for the cuda-checkpoint process %d", child_pid);
		goto err;
	}
	if (WIFSIGNALED(status)) {
		int sig = WTERMSIG(status);
		pr_err("cuda-checkpoint unexpectedly signaled with %d: %s\n", sig, strsignal(sig));
	} else if (WIFEXITED(status)) {
		exit_code = WEXITSTATUS(status);
	} else {
		pr_err("cuda-checkpoint exited improperly: %u\n", status);
	}

	if (exit_code != EXIT_SUCCESS)
		pr_debug("cuda-checkpoint output ===>\n%s\n"
			 "<=== cuda-checkpoint output\n",
			 buf);

	g_timing_stats.total_cuda_checkpoint_calls++;
	TIMING_END(launch_start, g_timing_stats.total_cuda_checkpoint_us);

	return exit_code;
err:
	kill(child_pid, SIGKILL);
	waitpid(child_pid, NULL, 0);
	return -1;
}

/**
 * Checks if a given flag is supported by the cuda-checkpoint utility
 *
 * Returns:
 *  1 if the flag is supported,
 *  0 if the flag is not supported,
 *  -1 if there was an error launching the cuda-checkpoint utility.
 */
static int cuda_checkpoint_supports_flag(const char *flag)
{
	char msg_buf[2048];
	const char *args[] = { CUDA_CHECKPOINT, "-h", NULL };

	if (launch_cuda_checkpoint(args, msg_buf, sizeof(msg_buf)) != 0)
		return -1;

	if (strstr(msg_buf, flag) == NULL)
		return 0;

	return 1;
}

/* Retrieve the cuda restore thread TID from the root pid */
static int get_cuda_restore_tid(int root_pid)
{
	char pid_buf[16];
	char pid_out[CUDA_CKPT_BUF_SIZE];

	snprintf(pid_buf, sizeof(pid_buf), "%d", root_pid);

	const char *args[] = { CUDA_CHECKPOINT, "--get-restore-tid", "--pid", pid_buf, NULL };
	int ret = launch_cuda_checkpoint(args, pid_out, sizeof(pid_out));
	if (ret != 0) {
		pr_err("Failed to launch cuda-checkpoint to retrieve restore tid: %s\n", pid_out);
		return -1;
	}

	return atoi(pid_out);
}

static cuda_task_state_t get_task_state_enum(const char *state_str)
{
	if (strncmp(state_str, "running", 7) == 0)
		return CUDA_TASK_RUNNING;

	if (strncmp(state_str, "locked", 6) == 0)
		return CUDA_TASK_LOCKED;

	if (strncmp(state_str, "checkpointed", 12) == 0)
		return CUDA_TASK_CHECKPOINTED;

	pr_err("Unknown CUDA state: %s\n", state_str);
	return CUDA_TASK_UNKNOWN;
}

static cuda_task_state_t get_cuda_state(pid_t pid)
{
	char pid_buf[16];
	char state_str[CUDA_CKPT_BUF_SIZE];
	const char *args[] = { CUDA_CHECKPOINT, "--get-state", "--pid", pid_buf, NULL };

	snprintf(pid_buf, sizeof(pid_buf), "%d", pid);

	if (launch_cuda_checkpoint(args, state_str, sizeof(state_str))) {
		pr_err("Failed to launch cuda-checkpoint to retrieve state: %s\n", state_str);
		return CUDA_TASK_UNKNOWN;
	}

	return get_task_state_enum(state_str);
}

static int cuda_process_checkpoint_action(int pid, const char *action, unsigned int timeout, char *msg_buf,
					  int buf_size)
{
	char pid_buf[16];
	char timeout_buf[16];

	snprintf(pid_buf, sizeof(pid_buf), "%d", pid);

	const char *args[] = { CUDA_CHECKPOINT, "--action", action, "--pid", pid_buf, NULL /* --timeout */,
			       NULL /* timeout_val */, NULL };
	if (timeout > 0) {
		snprintf(timeout_buf, sizeof(timeout_buf), "%d", timeout);
		args[5] = "--timeout";
		args[6] = timeout_buf;
	}

	return launch_cuda_checkpoint(args, msg_buf, buf_size);
}

static int interrupt_restore_thread(int restore_tid, k_rtsigset_t *restore_sigset)
{
	/* Since we resumed a thread that CRIU previously already froze we need to
	 * INTERRUPT it once again, task was already SEIZE'd so we don't need to do
	 * a compel_interrupt_task()
	 */
	if (ptrace(PTRACE_INTERRUPT, restore_tid, NULL, 0)) {
		pr_perror("Could not interrupt cuda restore tid %d after checkpoint, process may be in strange state",
			  restore_tid);
		return -1;
	}

	struct proc_status_creds creds;
	if (compel_wait_task(restore_tid, -1, parse_pid_status, NULL, &creds.s, NULL) != COMPEL_TASK_ALIVE) {
		pr_err("compel_wait_task failed after interrupt\n");
		return -1;
	}

	if (ptrace(PTRACE_SETOPTIONS, restore_tid, NULL, PTRACE_O_SUSPEND_SECCOMP | PTRACE_O_TRACESYSGOOD)) {
		pr_perror("Failed to set ptrace options on interrupt for restore tid %d", restore_tid);
		return -1;
	}

	if (ptrace(PTRACE_SETSIGMASK, restore_tid, sizeof(*restore_sigset), restore_sigset)) {
		pr_perror("Unable to restore original sigmask to restore tid %d", restore_tid);
		return -1;
	}

	return 0;
}

static int resume_restore_thread(int restore_tid, k_rtsigset_t *save_sigset)
{
	k_rtsigset_t block;

	if (ptrace(PTRACE_GETSIGMASK, restore_tid, sizeof(*save_sigset), save_sigset)) {
		pr_perror("Failed to get current sigmask for restore tid %d", restore_tid);
		return -1;
	}

	ksigfillset(&block);
	ksigdelset(&block, SIGTRAP);

	if (ptrace(PTRACE_SETSIGMASK, restore_tid, sizeof(block), &block)) {
		pr_perror("Failed to block signals on restore tid %d", restore_tid);
		return -1;
	}

	// Clear out PTRACE_O_SUSPEND_SECCOMP when we resume the restore thread
	if (ptrace(PTRACE_SETOPTIONS, restore_tid, NULL, 0)) {
		pr_perror("Could not clear ptrace options on restore tid %d", restore_tid);
		return -1;
	}

	if (ptrace(PTRACE_CONT, restore_tid, NULL, 0)) {
		pr_perror("Could not resume cuda restore tid %d", restore_tid);
		return -1;
	}

	return 0;
}

int cuda_plugin_checkpoint_devices(int pid)
{
	int restore_tid = -1;
	char msg_buf[CUDA_CKPT_BUF_SIZE];
	int int_ret;
	int status;
	k_rtsigset_t save_sigset;
	struct pid_info *task_info;
	bool pid_found = false;
	TIMING_START(checkpoint_start);

	if (plugin_disabled) {
		return -ENOTSUP;
	}

	/* Check if the process is already in a checkpointed state and get cached restore_tid */
	list_for_each_entry(task_info, &cuda_pids, list) {
		if (task_info->pid == pid) {
			if (task_info->initial_task_state == CUDA_TASK_CHECKPOINTED) {
				pr_info("pid %d already in a checkpointed state\n", pid);
				return 0;
			}
			restore_tid = task_info->restore_tid; /* Use cached restore TID */
			pid_found = true;
			pr_debug("Using cached restore_tid %d for pid %d\n", restore_tid, pid);
			break;
		}
	}

	if (pid_found == false) {
		/* Not found in cache, try to get it directly */
		restore_tid = get_cuda_restore_tid(pid);

		/* We can possibly hit a race with cuInit() where we are past the point of
		 * locking the process but at lock time cuInit() hadn't completed in which
		 * case cuda-checkpoint will report that we're in an invalid state to
		 * checkpoint
		 */
		if (restore_tid == -1) {
			pr_info("No need to checkpoint devices on pid %d\n", pid);
			return 0;
		}

		/* We return an error here. The task should be restored
		 * to its original state at cuda_plugin_fini().
		 */
		pr_err("Failed to track pid %d (not found in pause cache)\n", pid);
		return -1;
	}

	/* At this point we have a valid restore_tid from cache */
	if (restore_tid == -1) {
		pr_err("Invalid cached restore_tid for pid %d\n", pid);
		return -1;
	}

	pr_info("Checkpointing CUDA devices on pid %d restore_tid %d\n", pid, restore_tid);
	/* We need to resume the checkpoint thread to prepare the mappings for
	 * checkpointing
	 */
	if (resume_restore_thread(restore_tid, &save_sigset)) {
		return -1;
	}

	task_info->checkpointed = 1;
	status = cuda_process_checkpoint_action(pid, ACTION_CHECKPOINT, 0, msg_buf, sizeof(msg_buf));
	if (status) {
		pr_err("CHECKPOINT_DEVICES failed with %s\n", msg_buf);
	}

	int_ret = interrupt_restore_thread(restore_tid, &save_sigset);

	TIMING_END(checkpoint_start, g_timing_stats.checkpoint_devices_us);
	pr_info("CHECKPOINT_DEVICES on pid %d completed (cumulative time: %lu us)\n", pid,
		g_timing_stats.checkpoint_devices_us);

	return status != 0 ? -1 : int_ret;
}
CR_PLUGIN_REGISTER_HOOK(CR_PLUGIN_HOOK__CHECKPOINT_DEVICES, cuda_plugin_checkpoint_devices);

int cuda_plugin_pause_devices(int pid)
{
	int restore_tid;
	char msg_buf[CUDA_CKPT_BUF_SIZE];
	cuda_task_state_t task_state;
	TIMING_START(pause_start);

	if (plugin_disabled) {
		return -ENOTSUP;
	}

	restore_tid = get_cuda_restore_tid(pid);

	if (restore_tid == -1) {
		pr_info("no need to pause devices on pid %d\n", pid);
		return 0;
	}

	task_state = get_cuda_state(restore_tid);
	if (task_state == CUDA_TASK_UNKNOWN) {
		pr_err("Failed to get CUDA state for PID %d\n", restore_tid);
		return -1;
	}

	if (!plugin_added_to_inventory) {
		if (add_inventory_plugin(CR_PLUGIN_DESC.name)) {
			pr_err("Failed to add CUDA plugin to inventory image\n");
			return -1;
		}
		plugin_added_to_inventory = true;
	}

	if (task_state == CUDA_TASK_LOCKED) {
		pr_info("pid %d already in a locked state\n", pid);
		/* Leave this PID in a "locked" state at resume_device() */
		add_pid_to_buf(&cuda_pids, pid, CUDA_TASK_LOCKED, restore_tid);
		return 0;
	}

	if (task_state == CUDA_TASK_CHECKPOINTED) {
		/* We need to skip this PID in cuda_plugin_checkpoint_devices(),
		 * and leave it in a "checkpoined" state at resume_device(). */
		add_pid_to_buf(&cuda_pids, pid, CUDA_TASK_CHECKPOINTED, restore_tid);
		return 0;
	}

	pr_info("pausing devices on pid %d\n", pid);
	int status = cuda_process_checkpoint_action(pid, ACTION_LOCK, opts.timeout * 1000, msg_buf, sizeof(msg_buf));
	if (status) {
		pr_err("PAUSE_DEVICES failed with %s\n", msg_buf);
		if (alarm_timeouted())
			goto unlock;
		return -1;
	}

	if (add_pid_to_buf(&cuda_pids, pid, CUDA_TASK_RUNNING, restore_tid)) {
		pr_err("unable to track paused pid %d\n", pid);
		goto unlock;
	}

	TIMING_END(pause_start, g_timing_stats.pause_devices_us);
	pr_info("PAUSE_DEVICES on pid %d completed (cumulative time: %lu us)\n", pid, g_timing_stats.pause_devices_us);
	return 0;
unlock:
	status = cuda_process_checkpoint_action(pid, ACTION_UNLOCK, 0, msg_buf, sizeof(msg_buf));
	if (status) {
		pr_err("Failed to unlock process status %s, pid %d may hang\n", msg_buf, pid);
	}
	return -1;
}
CR_PLUGIN_REGISTER_HOOK(CR_PLUGIN_HOOK__PAUSE_DEVICES, cuda_plugin_pause_devices)

int resume_device(int pid, int checkpointed, cuda_task_state_t initial_task_state)
{
	char msg_buf[CUDA_CKPT_BUF_SIZE];
	int status;
	int ret = 0;
	int int_ret;
	k_rtsigset_t save_sigset;
	TIMING_START(resume_start);

	if (initial_task_state == CUDA_TASK_UNKNOWN) {
		pr_info("skip resume for PID %d (unknown state)\n", pid);
		return 0;
	}

	int restore_tid = get_cuda_restore_tid(pid);
	if (restore_tid == -1) {
		pr_info("No need to resume devices on pid %d\n", pid);
		return 0;
	}

	pr_info("resuming devices on pid %d\n", pid);
	/* The resuming process has to stay frozen during this time otherwise
	 * attempting to access a UVM pointer will crash if we haven't restored the
	 * underlying mappings yet
	 */
	pr_debug("Restore thread pid %d found for real pid %d\n", restore_tid, pid);
	/* wakeup the restore thread so we can handle the restore for this pid,
	 * rseq_cs has to be restored before execution
	 */
	if (resume_restore_thread(restore_tid, &save_sigset)) {
		return -1;
	}

	if (checkpointed && (initial_task_state == CUDA_TASK_RUNNING || initial_task_state == CUDA_TASK_LOCKED)) {
		/* If the process was "locked" or "running" before checkpointing it, we need to restore it */
		status = cuda_process_checkpoint_action(pid, ACTION_RESTORE, 0, msg_buf, sizeof(msg_buf));
		if (status) {
			pr_err("RESUME_DEVICES RESTORE failed with %s\n", msg_buf);
			ret = -1;
			goto interrupt;
		}
	}

	if (initial_task_state == CUDA_TASK_RUNNING) {
		/* If the process was "running" before we paused it, we need to unlock it */
		status = cuda_process_checkpoint_action(pid, ACTION_UNLOCK, 0, msg_buf, sizeof(msg_buf));
		if (status) {
			pr_err("RESUME_DEVICES UNLOCK failed with %s\n", msg_buf);
			ret = -1;
		}
	}

interrupt:
	int_ret = interrupt_restore_thread(restore_tid, &save_sigset);

	TIMING_END(resume_start, g_timing_stats.resume_devices_us);
	pr_info("RESUME_DEVICE on pid %d completed (cumulative time: %lu us)\n", pid, g_timing_stats.resume_devices_us);

	return ret != 0 ? ret : int_ret;
}

/* Option A: Check if a PID likely has CUDA based on checkpoint image data
 * This checks for CUDA indicators without needing the process to be initialized
 */
static bool pid_has_cuda_in_checkpoint(int pid)
{
	char path[PATH_MAX];
	FILE *fp;
	char line[512];
	bool has_cuda = false;

	/* Check for CUDA UVM socket in files.img
	 * CUDA tasks have unix sockets like @cuda-uvmfd-<namespace>-<pid>@
	 */
	snprintf(path, sizeof(path), "/proc/%d/net/unix", pid);
	fp = fopen(path, "r");
	if (fp) {
		while (fgets(line, sizeof(line), fp)) {
			if (strstr(line, "cuda-uvmfd")) {
				pr_err("OPTION_A: pid %d has cuda-uvmfd socket in /proc\n", pid);
				has_cuda = true;
				break;
			}
		}
		fclose(fp);
	}

	/* Also check for /dev/nvidia device mappings */
	if (!has_cuda) {
		snprintf(path, sizeof(path), "/proc/%d/maps", pid);
		fp = fopen(path, "r");
		if (fp) {
			while (fgets(line, sizeof(line), fp)) {
				if (strstr(line, "/dev/nvidia") || strstr(line, "libcuda")) {
					pr_err("OPTION_A: pid %d has nvidia device/lib mapping\n", pid);
					has_cuda = true;
					break;
				}
			}
			fclose(fp);
		}
	}

	return has_cuda;
}

/* Thread function for async GPU restore */
static void *async_restore_thread(void *arg)
{
	struct async_restore_task *task = (struct async_restore_task *)arg;

	pr_err("ASYNC_THREAD: starting restore for pid %d\n", task->pid);

	task->result = resume_device(task->pid, 1, CUDA_TASK_RUNNING);

	pthread_mutex_lock(&async_restore_lock);
	task->completed = true;
	pthread_mutex_unlock(&async_restore_lock);

	pr_err("ASYNC_THREAD: completed restore for pid %d (result: %d)\n", task->pid, task->result);

	return NULL;
}

/* POST_FORKING hook: Discover and start async GPU restore operations */
int cuda_plugin_post_forking(void)
{
	struct pstree_item *item;
	int ret = 0;
	int cuda_task_count = 0;

	pr_err("===== POST_FORKING HOOK CALLED =====\n");

	if (plugin_disabled) {
		pr_err("POST_FORKING: plugin_disabled=true, returning -ENOTSUP\n");
		return -ENOTSUP;
	}

	if (!async_restore_enabled) {
		pr_err("POST_FORKING: async_restore_enabled=false, skipping overlap (baseline mode)\n");
		return 0;
	}

	pr_err("POST_FORKING: Scanning for CUDA tasks to restore asynchronously\n");

	/* Iterate through all tasks and start async restore for those with CUDA */
	for_each_pstree_item(item) {
		struct async_restore_task *task;
		bool has_cuda_checkpoint;

		pr_err("POST_FORKING: Checking pstree_item real_pid=%d state=%d\n",
		       item->pid ? item->pid->real : -1,
		       item->pid ? item->pid->state : -1);

		if (!task_alive(item)) {
			pr_err("POST_FORKING: pid=%d not alive, skipping\n", item->pid ? item->pid->real : -1);
			continue;
		}

		/* OPTION A: Check if PID has CUDA indicators in /proc (checkpoint state) */
		has_cuda_checkpoint = pid_has_cuda_in_checkpoint(item->pid->real);
		pr_err("POST_FORKING: pid=%d checkpoint CUDA check = %d\n", item->pid->real, has_cuda_checkpoint);

		if (!has_cuda_checkpoint) {
			/* No CUDA indicators in checkpoint, skip */
			continue;
		}

		pr_err("POST_FORKING: pid=%d HAS CUDA, will start async restore\n", item->pid->real);

		/* Register and start async restore for this task */
		task = xmalloc(sizeof(*task));
		if (task == NULL) {
			pr_err("Failed to allocate async restore task for pid %d\n", item->pid->real);
			ret = -1;
			continue;
		}

		task->pid = item->pid->real;
		task->result = 0;
		task->started = false;
		task->completed = false;

		pthread_mutex_lock(&async_restore_lock);
		list_add_tail(&task->list, &async_restore_tasks);
		pthread_mutex_unlock(&async_restore_lock);

		pr_err("POST_FORKING: launching async restore thread for CUDA task pid %d\n", task->pid);

		if (pthread_create(&task->thread, NULL, async_restore_thread, task) != 0) {
			pr_perror("Failed to create async restore thread for pid %d", task->pid);
			ret = -1;
		} else {
			task->started = true;
			cuda_task_count++;
		}
	}

	pthread_mutex_lock(&async_restore_lock);
	if (cuda_task_count > 0) {
		async_restore_triggered = true;
	}
	pthread_mutex_unlock(&async_restore_lock);

	pr_err("POST_FORKING: Started async restore for %d CUDA tasks (triggered=%d)\n",
	       cuda_task_count, async_restore_triggered);

	return ret;
}
CR_PLUGIN_REGISTER_HOOK(CR_PLUGIN_HOOK__POST_FORKING, cuda_plugin_post_forking);

int cuda_plugin_resume_devices_late(int pid)
{
	struct async_restore_task *task;
	int ret = 0;

	if (plugin_disabled) {
		return -ENOTSUP;
	}

	/* OPTION B: If async restore wasn't triggered by POST_FORKING, OR if this task
	 * wasn't found by Option A, trigger async restore for remaining tasks.
	 * This catches tasks that were forked after POST_FORKING.
	 */
	if (!async_restore_enabled) {
		/* Baseline mode: no async restore, do everything synchronously */
		pr_info("RESUME_DEVICES_LATE: async disabled, doing synchronous restore for pid %d\n", pid);
		return resume_device(pid, 1, CUDA_TASK_RUNNING);
	}

	pthread_mutex_lock(&async_restore_lock);

	/* Check if current task is already in async list */
	bool current_task_in_async = false;
	list_for_each_entry(task, &async_restore_tasks, list) {
		if (task->pid == pid) {
			current_task_in_async = true;
			break;
		}
	}

	/* If not triggered yet, OR if current task wasn't found by Option A, scan for more */
	if (!async_restore_triggered || !current_task_in_async) {
		struct pstree_item *item;
		int cuda_task_count = 0;

		pr_err("OPTION_B: First CUDA task (pid %d) triggering async restore for other tasks\n", pid);
		async_restore_triggered = true;
		pthread_mutex_unlock(&async_restore_lock);

		/* Scan all tasks and start async restore for CUDA tasks (INCLUDING this one) */
		for_each_pstree_item(item) {
			int restore_tid;
			struct async_restore_task *new_task;
			bool already_in_list = false;

			if (!task_alive(item))
				continue;

			/* Check if this task is already in async list (from Option A) */
			pthread_mutex_lock(&async_restore_lock);
			list_for_each_entry(task, &async_restore_tasks, list) {
				if (task->pid == item->pid->real) {
					already_in_list = true;
					break;
				}
			}
			pthread_mutex_unlock(&async_restore_lock);

			if (already_in_list) {
				pr_err("OPTION_B: pid %d already in async list, skipping\n", item->pid->real);
				continue;
			}

			/* Check if this task has CUDA */
			restore_tid = get_cuda_restore_tid(item->pid->real);
			if (restore_tid == -1) {
				continue;
			}

			pr_err("OPTION_B: Found NEW CUDA task pid=%d, starting async restore\n", item->pid->real);

			/* Start async restore for this task */
			new_task = xmalloc(sizeof(*new_task));
			if (new_task == NULL) {
				pr_err("OPTION_B: Failed to allocate async task for pid %d\n", item->pid->real);
				continue;
			}

			new_task->pid = item->pid->real;
			new_task->result = 0;
			new_task->started = false;
			new_task->completed = false;

			pthread_mutex_lock(&async_restore_lock);
			list_add_tail(&new_task->list, &async_restore_tasks);
			pthread_mutex_unlock(&async_restore_lock);

			if (pthread_create(&new_task->thread, NULL, async_restore_thread, new_task) != 0) {
				pr_perror("OPTION_B: Failed to create async thread for pid %d", item->pid->real);
			} else {
				new_task->started = true;
				cuda_task_count++;
			}
		}

		pr_err("OPTION_B: Started async restore for %d additional CUDA tasks\n", cuda_task_count);
		pthread_mutex_lock(&async_restore_lock);
	}

	/* Wait for async restore to complete if it was started */
	list_for_each_entry(task, &async_restore_tasks, list) {
		if (task->pid == pid) {
			pthread_mutex_unlock(&async_restore_lock);

			if (task->started) {
				pr_info("RESUME_DEVICES_LATE: waiting for async restore of pid %d\n", pid);
				pthread_join(task->thread, NULL);
				ret = task->result;
				pr_info("RESUME_DEVICES_LATE: async restore of pid %d completed with result %d\n", pid,
					ret);
			} else {
				/* Async restore wasn't started, do it synchronously */
				pr_info("RESUME_DEVICES_LATE: performing synchronous restore of pid %d\n", pid);
				ret = resume_device(pid, 1, CUDA_TASK_RUNNING);
			}

			return ret;
		}
	}
	pthread_mutex_unlock(&async_restore_lock);

	/* If no async task was registered, do synchronous restore */
	pr_info("RESUME_DEVICES_LATE: no async task found for pid %d, doing synchronous restore\n", pid);
	return resume_device(pid, 1, CUDA_TASK_RUNNING);
}
CR_PLUGIN_REGISTER_HOOK(CR_PLUGIN_HOOK__RESUME_DEVICES_LATE, cuda_plugin_resume_devices_late)

/**
 * Check if a CUDA device is available on the system
 */
static bool is_cuda_device_available(void)
{
	const char *gpu_path = "/proc/driver/nvidia/gpus/";
	struct stat sb;

	if (stat(gpu_path, &sb) != 0)
		return false;

	return S_ISDIR(sb.st_mode);
}

int cuda_plugin_init(int stage)
{
	int ret;

	/* Disable CUDA checkpointing with pre-dump */
	if (stage == CR_PLUGIN_STAGE__PRE_DUMP) {
		plugin_disabled = true;
		return 0;
	}

	if (stage == CR_PLUGIN_STAGE__RESTORE) {
		if (!check_and_remove_inventory_plugin(CR_PLUGIN_DESC.name, strlen(CR_PLUGIN_DESC.name))) {
			plugin_disabled = true;
			return 0;
		}
	}

	if (!fault_injected(FI_PLUGIN_CUDA_FORCE_ENABLE) && !is_cuda_device_available()) {
		pr_info("No GPU device found; CUDA plugin is disabled\n");
		plugin_disabled = true;
		return 0;
	}

	ret = cuda_checkpoint_supports_flag("--action");
	if (ret == -1) {
		pr_warn("check that %s is present in $PATH\n", CUDA_CHECKPOINT);
		plugin_disabled = true;
		return 0;
	}

	if (ret == 0) {
		pr_warn("cuda-checkpoint --action flag not supported, an r555 or higher version driver is required. Disabling CUDA plugin\n");
		plugin_disabled = true;
		return 0;
	}

	pr_info("initialized: %s stage %d\n", CR_PLUGIN_DESC.name, stage);

	/* In the DUMP stage track all the PID's we've paused CUDA operations on to
	 * release them when we're done if the user requested the leave-running option
	 */
	if (stage == CR_PLUGIN_STAGE__DUMP) {
		INIT_LIST_HEAD(&cuda_pids);
	}

	/* In the RESTORE stage, initialize async restore task list */
	if (stage == CR_PLUGIN_STAGE__RESTORE) {
		INIT_LIST_HEAD(&async_restore_tasks);
	}

	set_compel_interrupt_only_mode();

	return 0;
}

void cuda_plugin_fini(int stage, int ret)
{
	if (plugin_disabled) {
		return;
	}

	pr_info("finished %s stage %d err %d\n", CR_PLUGIN_DESC.name, stage, ret);

	/* Print timing summary */
	pr_info("==== CUDA Plugin Timing Summary ====\n");
	pr_info("  PAUSE_DEVICES:      %lu us\n", g_timing_stats.pause_devices_us);
	pr_info("  CHECKPOINT_DEVICES: %lu us\n", g_timing_stats.checkpoint_devices_us);
	pr_info("  RESUME_DEVICES:     %lu us\n", g_timing_stats.resume_devices_us);
	pr_info("  cuda-checkpoint calls: %lu (total time: %lu us)\n", g_timing_stats.total_cuda_checkpoint_calls,
		g_timing_stats.total_cuda_checkpoint_us);
	pr_info("  TOTAL PLUGIN TIME:  %lu us (%.3f seconds)\n",
		g_timing_stats.pause_devices_us + g_timing_stats.checkpoint_devices_us +
			g_timing_stats.resume_devices_us,
		(g_timing_stats.pause_devices_us + g_timing_stats.checkpoint_devices_us +
		 g_timing_stats.resume_devices_us) /
			1000000.0);
	pr_info("====================================\n");

	/* Release all the paused PID's at the end of the DUMP stage in case the
	 * user provides the -R (leave-running) flag or an error occurred
	 */
	if (stage == CR_PLUGIN_STAGE__DUMP && (opts.final_state == TASK_ALIVE || ret != 0)) {
		struct pid_info *info;
		list_for_each_entry(info, &cuda_pids, list) {
			resume_device(info->pid, info->checkpointed, info->initial_task_state);
		}
	}
	if (stage == CR_PLUGIN_STAGE__DUMP) {
		dealloc_pid_buffer(&cuda_pids);
	}

	if (stage == CR_PLUGIN_STAGE__RESTORE) {
		dealloc_async_restore_tasks();
	}
}
CR_PLUGIN_REGISTER("cuda_plugin", cuda_plugin_init, cuda_plugin_fini)

