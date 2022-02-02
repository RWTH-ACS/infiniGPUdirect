#include "vector.h"

int vector_total(vector_t *v)
{
    int totalCount = UNDEFINE;
    if(v)
    {
        totalCount = v->vector_list.total;
    }
    return totalCount;
}

int vector_resize(vector_t *v, int capacity)
{
    int  status = UNDEFINE;
    if(v)
    {
        void **items = realloc(v->vector_list.items, sizeof(void *) * capacity);
        if (items)
        {
            v->vector_list.items = items;
            v->vector_list.capacity = capacity;
            status = SUCCESS;
        }
    }
    return status;
}

int vector_push_back(vector_t *v, void *item)
{
    int  status = UNDEFINE;
    if(v)
    {
        if (v->vector_list.capacity == v->vector_list.total)
        {
            status = vector_resize(v, v->vector_list.capacity * 2);
            if(status != UNDEFINE)
            {
                v->vector_list.items[v->vector_list.total++] = item;
            }
        }
        else
        {
            v->vector_list.items[v->vector_list.total++] = item;
            status = SUCCESS;
        }
    }
    return status;
}

int vector_set(vector_t *v, int index, void *item)
{
    int  status = UNDEFINE;
    if(v)
    {
        if ((index >= 0) && (index < v->vector_list.total))
        {
            v->vector_list.items[index] = item;
            status = SUCCESS;
        }
    }
    return status;
}

void *vector_get(vector_t *v, int index)
{
    void *readData = NULL;
    if(v)
    {
        if ((index >= 0) && (index < v->vector_list.total))
        {
            readData = v->vector_list.items[index];
        }
    }
    return readData;
}

int vector_delete(vector_t *v, int index)
{
    int  status = UNDEFINE;
    int i = 0;
    if(v)
    {
        if ((index < 0) || (index >= v->vector_list.total))
            return status;
        v->vector_list.items[index] = NULL;
        for (i = index; (i < v->vector_list.total - 1); ++i)
        {
            v->vector_list.items[i] = v->vector_list.items[i + 1];
            v->vector_list.items[i + 1] = NULL;
        }
        v->vector_list.total--;
        if ((v->vector_list.total > 0) && ((v->vector_list.total) == (v->vector_list.capacity / 4)))
        {
            vector_resize(v, v->vector_list.capacity / 2);
        }
        status = SUCCESS;
    }
    return status;
}

int vector_free(vector_t *v)
{
    int  status = UNDEFINE;
    if(v)
    {
        free(v->vector_list.items);
        v->vector_list.items = NULL;
        status = SUCCESS;
    }
    return status;
}

void vector_init(vector_t *v)
{
    //init function pointers
    v->vector_total = vector_total;
    v->vector_resize = vector_resize;
    v->vector_add = vector_push_back;
    v->vector_set = vector_set;
    v->vector_get = vector_get;
    v->vector_free = vector_free;
    v->vector_delete = vector_delete;
    //initialize the capacity and allocate the memory
    v->vector_list.capacity = VECTOR_INIT_CAPACITY;
    v->vector_list.total = 0;
    v->vector_list.items = malloc(sizeof(void *) * v->vector_list.capacity);
}