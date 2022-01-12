#include <stdio.h>
#include <stdlib.h>

#define VECTOR_INIT_CAPACITY 6
#define UNDEFINE  -1
#define SUCCESS 0
 
//Store and track the stored data
typedef struct vector_list
{
    void **items;
    int capacity;
    int total;
} vector_list;
//structure contain the function pointer
typedef struct vector vector_t;
struct vector
{
    vector_list vector_list;
//function pointers
    int (*vector_total)(vector_t *);
    int (*vector_resize)(vector_t *, int);
    int (*vector_add)(vector_t *, void *);
    int (*vector_set)(vector_t *, int, void *);
    void *(*vector_get)(vector_t *, int);
    int (*vector_delete)(vector_t *, int);
    int (*vector_free)(vector_t *);
};


int vector_total(vector_t *v);
int vector_resize(vector_t *v, int capacity);
int vector_push_back(vector_t *v, void *item);
int vector_set(vector_t *v, int index, void *item);
void *vector_get(vector_t *v, int index);
int vector_delete(vector_t *v, int index);
int vector_free(vector_t *v);
void vector_init(vector_t *v);