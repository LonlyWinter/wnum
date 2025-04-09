#include <stdio.h>
#include <math.h>

#include "utils.cuh"


__device__ size_t get_id() {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    return id;
}


__device__ float add(const float x, const float y) {
    return x + y;
}

__device__ float sub(const float x, const float y) {
    return x - y;
}

__device__ float mul(const float x, const float y) {
    return x * y;
}

__device__ float div(const float x, const float y) {
    return x / y;
}


__global__ void op_map_item_broadcast(float* A, const float B, float* C, const size_t N, const u_int8_t op) {
    const size_t id = get_id();

    if (id >= N) return;

    switch (op) {
        case 0:
            // printf("CPP CUDA Add: %2d, %2f + %2f\n", id, A[id], B[id]);
            C[id] = add(A[id], B);
            break;

        case 1:
            C[id] = sub(A[id], B);
            break;

        case 2:
            C[id] = mul(A[id], B);
            break;

        case 3:
            C[id] = div(A[id], B);
            break;

        case 4: {
            const float factor = __powf(10.0, B);
            C[id] = roundf(A[id] * factor) / factor;
            break;
        }
        
        default:
            break;
    }
}


__global__ void op_map_item(float* A, float* B, float* C, const size_t N, const u_int8_t op) {
    const size_t id = get_id();

    if (id >= N) return;

    switch (op) {
        case 0:
            // printf("CPP CUDA Add: %2d, %2f + %2f\n", id, A[id], B[id]);
            C[id] = add(A[id], B[id]);
            break;

        case 1:
            C[id] = sub(A[id], B[id]);
            break;

        case 2:
            C[id] = mul(A[id], B[id]);
            break;

        case 3:
            C[id] = div(A[id], B[id]);
            break;

        case 4:
            if (A[id] == B[id]) {
                C[id] = 1.0;
            } else {
                C[id] = 0.0;
            }
            break;
        
        default:
            break;
    }
}


__global__ void op_map_item_single(float* A, float* C, const size_t N, const u_int8_t op) {
    const size_t id = get_id();

    if (id >= N) return;

    switch (op) {
        case 0:
            C[id] = __expf(A[id]);
            break;

        case 1:
            C[id] = __log2f(A[id]);
            break;

        case 2:
            C[id] = __logf(A[id]);
            break;

        case 3:
            C[id] = -A[id];
            break;

        case 4:
            if (A[id] > 0) {
                C[id] = A[id];
            } else {
                C[id] = -A[id];
            }
            break;
        
        case 5:
            if (A[id] > 0) {
                C[id] = A[id];
            } else {
                C[id] = 0.0;
            }
            break;
        
        default:
            break;
    }
}


__global__ void op_map_item_single_where_cond(float* A, float* C, const size_t N, const float s, const float a, const float b) {
    
    const size_t id = get_id();

    if (id >= N) return;

    if (A[id] >= s) {
        C[id] = a;
    } else {
        C[id] = b;
    }
}


__global__ void f32_eq(float* A, float* B, bool* result, const size_t N) {
    const size_t id = get_id();

    if (id >= N || (!result)) return;

    if (A[id] != B[id]) {
        *result = false;
    }
}


__global__ void op_map_dim(float* A, float* C, const size_t N, const size_t dim0, const size_t n0, const size_t n1, const u_int8_t op) {
    const size_t id = get_id();
    
    if (id >= N) return;

    const size_t n0_temp = N / n0;
    const size_t n1_temp = N - n0 * n0_temp;
    const size_t i_start0 = n0_temp * n1 * dim0;

    float res = 0.0;
    size_t index;
    switch (op) {
        case 0:
            // mean
            for (size_t i = 0; i < dim0; i++) {
                index = n1_temp + i * n1 + i_start0;
                res += A[index];
            }
            res /= (float)dim0;
            break;

        case 1:
            // sum
            for (size_t i = 0; i < dim0; i++) {
                index = n1_temp + i * n1 + i_start0;
                res += A[index];
            }
            break;

        case 2:
            // max
            index = n1_temp + i_start0;
            res = A[index];
            for (size_t i = 1; i < dim0; i++) {
                index = n1_temp + i * n1 + i_start0;
                if (A[index] > res) {
                    res = A[index];
                } 
            }
            break;

        case 3:
            // min
            index = n1_temp + i_start0;
            res = A[index];
            for (size_t i = 1; i < dim0; i++) {
                index = n1_temp + i * n1 + i_start0;
                if (A[index] < res) {
                    res = A[index];
                } 
            }
            break;
        
        default:
            break;
    }
    C[id] = res;
}


__global__ void op_transpose(float* A, float* C, const size_t N, const size_t dim0, const size_t dim1, const size_t n0, const size_t n1) {
    const size_t id = get_id();
    
    if (id >= N) return;

    const size_t n0_temp = N / n0;
    const size_t n1_temp = N - n0 * n0_temp;
    const size_t i_start0 = n0_temp * dim0 * dim1 * n1;

    size_t a, b;
    for (size_t i = 0; i < dim0; i++) {
        for (size_t j = 0; j < dim1; j++) {
            a = i * dim1 * n1 + j * n1 + n1_temp + i_start0;
            b = j * dim0 * n1 + i * n1 + n1_temp + i_start0;
            C[b] = A[a];
        }
    }
}


__global__ void op_concat(float* A, float* B, float* C, const size_t N, const size_t dim_sum, const size_t dim0, const size_t dim1, const size_t n0, const size_t n1) {
    const size_t id = get_id();
    
    if (id >= N) return;

    const size_t n0_temp = N / n0;
    const size_t n1_temp = N - n0 * n0_temp;
    
    const size_t n0_start = n0_temp * n1;
    const size_t n0_start0 = n0_start * dim0;
    const size_t n0_start1 = n0_start * dim1;

    size_t a, b;
    for (size_t i = 0; i < dim_sum; i++) {
        b = n0_temp * dim_sum * n1 + i * n1 + n1_temp;
        if (i < dim0) {
            a = i * n1 + n0_start0 + n1_temp;
            C[b] = A[a];
        } else {
            a = (i - dim0) * n1 + n0_start1 + n1_temp;
            C[b] = B[a];
        }
    }
}


__global__ void op_broadcast(float* A, float* C, const size_t N, const size_t n, const size_t n0, const size_t n1) {
    const size_t id = get_id();
    
    if (id >= N) return;

    const size_t n0_temp = N / n0;
    const size_t n1_temp = N - n0 * n0_temp;
    
    const size_t i_start = n0_temp * n1;

    size_t a, b;
    for (size_t i = 0; i < n; i++) {
        b = n0_temp * n * n1 + i * n1 + n1_temp;
        a = i * n1 + i_start + n1_temp;
        C[b] = A[a];
    }
}


__global__ void op_matmul_1d(float* A, float* B, float* C, const size_t N) {
    const size_t id = get_id();
    
    if (id >= N) return;
    
    C[id] += mul(A[id], B[id]);
}


__global__ void op_matmul_2d(float* A, float* B, float* C, const size_t N, const size_t n, const size_t n0, const size_t n1) {
    const size_t id = get_id();
    
    if (id >= N) return;

    const size_t n0_temp = N / n0;
    const size_t n1_temp = N - n0 * n0_temp;
    const size_t i0 = n0_temp * n;
    
    float res = 0.0;
    size_t a, b;
    for (size_t i = 0; i < n; i++) {
        a = i0 + i;
        b = n1_temp + i * n1;
        res += A[a] * B[b];
    }
    C[id] = res;
}


__global__ void op_matmul_nd(float* A, float* B, float* C, const size_t N, const size_t n, const size_t n0, const size_t n1, const size_t pre) {
    const size_t id = get_id();
    
    if (id >= N) return;

    const size_t pre_temp = N / n0 / n1;
    const size_t pre_temp_left = N - pre_temp * n0 * n1;
    const size_t n0_temp = pre_temp_left / n0;
    const size_t n1_temp = pre_temp_left - n0_temp * n0;
    
    size_t start1 = pre_temp * n0 * n + n0_temp * n1;
    size_t start2 = pre_temp * n * n1 + n1_temp;
    
    float res = 0.0;
    size_t a, b;
    for (size_t i = 0; i < n; i++) {
        a = start1 + i;
        b = start2 + i * n1;
        res += A[a] * B[b];
    }
    C[id] = res;
}












struct F32 {
    float* data;
    size_t len;

    F32(size_t len_now) {
        size_t len_bytes = len_now * sizeof(float);
        cudaError_t err = cudaMalloc(&data, len_bytes);
        if (err == cudaSuccess) {
            len = len_now;
        } else {
            printf("CUDA Error When New With Len: %2d, %2d, %s\n", len_now, err, cudaGetErrorString(err));
            len = 0;
        }
    }

    F32(float* data_now, size_t len_now) {
        size_t len_bytes = len_now * sizeof(float);

        // printf("CUDA mallocing\n");
        cudaError_t err = cudaMalloc(&data, len_bytes);
        if (err == cudaSuccess) {
            // printf("CUDA copying\n");
            cudaError_t err1 = cudaMemcpy(data, data_now, len_bytes, cudaMemcpyHostToDevice);
            // printf("CUDA copy to device error: %d\n", err1);
            len = len_now;
        } else {
            // printf("CUDA malloc error\n");
            len = 0;
        }
    }
};

extern "C" {
    F32* wdata_new(float* data_now, size_t len_now) {
        // printf("CPP new F32 ...\n");
        F32* res = new F32(data_now, len_now);
        // printf("CPP new F32: %d\n", res->gpu);
        return res;
    }

    F32* wdata_new_with(F32* data_now, size_t len_now, size_t len_all) {
        F32* res = new F32(len_all);
        if (res->len == 0) {
            return res;
        }

        size_t prev = 0;
        for (size_t i = 0; i < len_now; i++) {
            float* dst = res->data + prev;
            size_t len_bytes = data_now[i].len * sizeof(float);
            prev += data_now[i].len;
            cudaMemcpy(
                dst,
                data_now[i].data,
                len_bytes,
                cudaMemcpyDeviceToDevice
            );
        }
        return res;
    }

    F32* wdata_clone(F32* data_now) {
        F32* res = new F32(data_now->len);
        if (res->len == 0) {
            return res;
        }

        size_t len_bytes = data_now->len * sizeof(float);
        cudaMemcpy(
            res->data,
            data_now->data,
            len_bytes,
            cudaMemcpyDeviceToDevice
        );
        return res;
    }

    void wdata_drop(F32* obj) {
        // printf("F32 dropping\n");
        // printf("F32 dropping device\n");
        cudaError_t err = cudaFree(obj->data);
        // printf("F32 dropped device: %d\n", err);
        // delete obj;
        // printf("F32 dropped\n");
    }

    u_int16_t device_error() {
        cudaError_t err = cudaGetLastError();
        return (u_int16_t)err;
    }

    void device_reset() {
        cudaDeviceReset();
    }

    void device_set(u_int8_t dev) {
        set_gpu((int)dev);
    }

    float wdata_get_data_single(const F32* obj, const size_t index) {
        size_t len_byte_single = sizeof(float);
        float res;
        float* src = obj->data + index;
        cudaMemcpy(&res, src, len_byte_single, cudaMemcpyDeviceToHost);
        return res;
    }

    float* wdata_get_data_with_index(const F32* obj, const size_t* index, const size_t len) {
        size_t len_byte_single = sizeof(float);
        size_t len_bytes = len * len_byte_single;
        float* res = (float*)malloc(len_bytes);
        for(int i = 0; i < len; i++) {
            float* dst = res + i;
            float* src = obj->data + index[i];
            cudaMemcpy(dst, src, len_byte_single, cudaMemcpyDeviceToHost);
        }
        return res;
    }

    float* wdata_get_data_all(const F32* obj) {
        size_t len_bytes = obj->len * sizeof(float);
        float* res = (float*)malloc(len_bytes);
        cudaError_t err1 = cudaMemcpy(res, obj->data, len_bytes, cudaMemcpyDeviceToHost);
        // printf("CUDA copy to host error: %2d, %2f\n", err1, h[55]);
        return res;
    }

    F32* wdata_basic_op(const F32* lhs, const F32* rhs, const u_int8_t op) {
        F32* res = new F32(lhs->len);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (lhs->len + block.x - 1) / 32;
        dim3 grid(g);

        op_map_item<<<grid, block>>>(
            lhs->data,
            rhs->data,
            res->data,
            lhs->len,
            op
        );
        
        cudaDeviceSynchronize();
        return res;
    }

    F32* wdata_broadcast_op(const F32* lhs, const float rhs, const u_int8_t op) {
        F32* res = new F32(lhs->len);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (lhs->len + block.x - 1) / 32;
        dim3 grid(g);

        op_map_item_broadcast<<<grid, block>>>(
            lhs->data,
            rhs,
            res->data,
            lhs->len,
            op
        );
        
        cudaDeviceSynchronize();
        return res;
    }

    F32* wdata_map_op(const F32* lhs, const u_int8_t op) {
        F32* res = new F32(lhs->len);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (lhs->len + block.x - 1) / 32;
        dim3 grid(g);

        op_map_item_single<<<grid, block>>>(
            lhs->data,
            res->data,
            lhs->len,
            op
        );
        
        cudaDeviceSynchronize();
        return res;
    }

    bool wdata_eq(const F32* lhs, const F32* rhs) {
        bool result_host = true;
        bool* result;
        size_t len_bytes = sizeof(bool);
        cudaMalloc(&result, len_bytes);
        cudaMemcpy(result, &result_host, len_bytes, cudaMemcpyHostToDevice);
        
        dim3 block(32);
        int g = (lhs->len + block.x - 1) / 32;
        dim3 grid(g);

        f32_eq<<<grid, block>>>(
            lhs->data,
            rhs->data,
            result,
            lhs->len
        );

        cudaMemcpy(&result_host, result, len_bytes, cudaMemcpyDeviceToDevice);

        cudaFree(result);

        return result_host;
    }

    F32* wdata_dim_op(const F32* obj, const size_t dim0, const size_t n0, const size_t n1, const u_int8_t op) {
        size_t len = n0 * n1;
        F32* res = new F32(len);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (len + block.x - 1) / 32;
        dim3 grid(g);

        op_map_dim<<<grid, block>>>(
            obj->data,
            res->data,
            len,
            dim0,
            n0,
            n1,
            op
        );
        
        cudaDeviceSynchronize();
        return res;
    }

    F32* wdata_transpose(const F32* obj, const size_t dim0, const size_t dim1, const size_t n0, const size_t n1) {
        F32* res = new F32(obj->len);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (obj->len + block.x - 1) / 32;
        dim3 grid(g);

        op_transpose<<<grid, block>>>(
            obj->data,
            res->data,
            obj->len,
            dim0,
            dim1,
            n0,
            n1
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %2d\n", err);
            res->len = 0;
        }
        
        cudaDeviceSynchronize();
        return res;
    }

    F32* wdata_concat(const F32* lhs, const F32* rhs, const size_t dim0, const size_t dim1, const size_t n0, const size_t n1) {
        const size_t dim_sum = dim0 + dim1;
        const size_t len = n0 * n1;
        F32* res = new F32(len * dim_sum);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (len + block.x - 1) / 32;
        dim3 grid(g);

        op_concat<<<grid, block>>>(
            lhs->data,
            rhs->data,
            res->data,
            len,
            dim_sum,
            dim0,
            dim1,
            n0,
            n1
        );
        
        cudaDeviceSynchronize();
        return res;
    }

    F32* wdata_broadcast(const F32* obj, const size_t n, const size_t n0, const size_t n1) {
        const size_t len = n0 * n1;
        F32* res = new F32(len * n);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (len + block.x - 1) / 32;
        dim3 grid(g);

        op_broadcast<<<grid, block>>>(
            obj->data,
            res->data,
            len,
            n,
            n0,
            n1
        );
        
        cudaDeviceSynchronize();
        return res;
    }


    F32* wdata_where_cond(const F32* obj, const float s, const float a, const float b) {
        F32* res = new F32(obj->len);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (obj->len + block.x - 1) / 32;
        dim3 grid(g);

        op_map_item_single_where_cond<<<grid, block>>>(
            obj->data,
            res->data,
            obj->len,
            s,
            a,
            b
        );

        
        
        cudaDeviceSynchronize();
        return res;
    }


    F32* wdata_matmul_1d(const F32* lhs, const F32* rhs) {
        F32* res_raw = wdata_basic_op(lhs, rhs, 0);
        float* data = wdata_get_data_all(res_raw);

        float data_res = 0;
        for(size_t i = 0; i < lhs->len; i++) {
            data_res += data[i];
        }
        free(data);
        wdata_drop(res_raw);
        return new F32(&data_res, 1);
    }

    F32* wdata_matmul_2d(const F32* lhs, const F32* rhs, const size_t n, const size_t n0, const size_t n1) {
        size_t len = n0 * n1;
        F32* res = new F32(len);
        if (res->len == 0) {
            cudaError_t err = cudaGetLastError();
            printf("CUDA Error When New: Err: %2d, lhs.len: %2d, rhs.len: %2d, n0: %2d, n1: %2d\n", err, lhs->len, rhs->len);
            return res;
        }
        
        dim3 block(32);
        int g = (len + block.x - 1) / 32;
        dim3 grid(g);

        op_matmul_2d<<<grid, block>>>(
            lhs->data,
            rhs->data,
            res->data,
            len,
            n,
            n0,
            n1
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %2d\n", err);
            res->len = 0;
        }
        
        cudaDeviceSynchronize();
        return res;
    }

    F32* wdata_matmul_nd(const F32* lhs, const F32* rhs, const size_t n, const size_t n0, const size_t n1, const size_t pre) {
        size_t len = pre * n0 * n1;
        F32* res = new F32(len);
        if (res->len == 0) {
            return res;
        }
        
        dim3 block(32);
        int g = (len + block.x - 1) / 32;
        dim3 grid(g);

        op_matmul_nd<<<grid, block>>>(
            lhs->data,
            rhs->data,
            res->data,
            len,
            n,
            n0,
            n1,
            pre
        );
        
        cudaDeviceSynchronize();
        return res;
    }
}
