// Minimal newlib hooks: stdout/stderr to the llm_core console, heap in DDR.
#include <errno.h>
#include <stddef.h>
#include <sys/stat.h>
#include "llm_hw.h"

int _write(int fd, const char *buf, int len) {
    (void)fd;
    for (int i = 0; i < len; i++) {
        while (LLM_STATUS & ST_CONS_FULL) {}
        LLM_CONSOLE = (uint8_t)buf[i];
    }
    return len;
}

void *_sbrk(ptrdiff_t incr) {
    static char *brk = (char *)HEAP_BASE;
    if (brk + incr > (char *)(HEAP_BASE + HEAP_SIZE) || brk + incr < (char *)HEAP_BASE) {
        errno = ENOMEM;
        return (void *)-1;
    }
    char *old = brk;
    brk += incr;
    return old;
}

int _read(int fd, char *buf, int len) { (void)fd; (void)buf; (void)len; return 0; }
int _close(int fd) { (void)fd; return -1; }
int _lseek(int fd, int off, int whence) { (void)fd; (void)off; (void)whence; return 0; }
int _fstat(int fd, struct stat *st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd) { (void)fd; return 1; }
int _getpid(void) { return 1; }
int _kill(int pid, int sig) { (void)pid; (void)sig; errno = EINVAL; return -1; }
void _exit(int code) { (void)code; for (;;) {} }
