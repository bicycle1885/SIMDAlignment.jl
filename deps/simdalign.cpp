#include "simdalign.h"

// 16-byte-aligned memory allocation (copied from Julia src/gc.c)
static void *malloc_a32(size_t sz)
{
    void *ptr;
    if (posix_memalign(&ptr, 32, sz))
        return NULL;
    return ptr;
}

buffer_t* make_buffer(void)
{
    buffer_t* buffer = (buffer_t*)malloc(sizeof(buffer_t));
    buffer->data = NULL;
    buffer->len = 0;
    return buffer;
}

int expand_buffer(buffer_t* buffer, size_t sz)
{
    if (buffer->len >= sz)
        return 0;
    void* buf = malloc_a32(sz);
    if (buf == NULL)
        return 1;
    free(buffer->data);
    buffer->data = buf;
    buffer->len = sz;
    return 0;
}

void free_buffer(buffer_t* buffer)
{
    free(buffer->data);
    free(buffer);
}
