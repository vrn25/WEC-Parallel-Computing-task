#include<stdio.h>
#include<stdlib.h>
#define CEIL(a, b) ((a-1)/b +1)
const int ROWS = 30;
const int COLUMNS = 30;
const int BLOCK_SIZE = 1024;
const int MATRIX_BYTES = ROWS*COLUMNS*sizeof(int);

__global__ void get_matrix(int *d_in, int *d_out){
  
  int idx = threadIdx.x;
  
  int m = ROWS>COLUMNS?COLUMNS:ROWS;
  int n = ROWS>COLUMNS?ROWS:COLUMNS;
  m--;n--;
  int n_iter = m+n-1;
  int tbc,left,up,corner;

  for(int i=0;i<n_iter;i++){
    int x_index,y_index;
    if(i >= n){
      // FOR THIRD PART OF ITERATIONS
      
      if(idx >= i-n+1){
        if(ROWS<=COLUMNS){
            x_index = i+1-idx+m-n;
            y_index = idx+1+n-m;
        }
        else{
            x_index = idx+1+n-m;
            y_index = i+1-idx+m-n;
        }
        tbc = *(d_in + ((x_index) * COLUMNS + (y_index)));
        if(!tbc){
          *(d_out + ((x_index) * COLUMNS + (y_index))) = 0;
        }
        else{
          left = *(d_out + ((x_index) * COLUMNS + (y_index-1)));
          up = *(d_out + ((x_index-1) * COLUMNS + (y_index)));
          corner = *(d_out + ((x_index-1) * COLUMNS + (y_index-1)));
          int mini = (left>up?up:left);
          *(d_out + ((x_index) * COLUMNS + (y_index))) = (mini>corner?corner:mini)+1;
        }
      }
    }
    else if(i>=m-1 && i<=n-1){
      // FOR SECOND PART OF ITERATIONS

      if(ROWS<=COLUMNS){
          x_index = m-idx;
          y_index = idx+i-m+2;
      }
      else{
          x_index = idx+i-m+2;
          y_index = m-idx;
      }
        tbc = *(d_in + ((x_index) * COLUMNS + (y_index)));
          if(!tbc)
            *(d_out + ((x_index) * COLUMNS + (y_index))) = 0;
          else{
            left = *(d_out + ((x_index) * COLUMNS + (y_index-1)));
            up = *(d_out + ((x_index-1) * COLUMNS + (y_index)));
            corner = *(d_out + ((x_index-1) * COLUMNS + (y_index-1)));
            int mini = (left>up?up:left);
            *(d_out + ((x_index) * COLUMNS + (y_index))) = (mini>corner?corner:mini)+1;
          }
    }
    else{
      // FOR FIRST PART OF ITERATIONS
      if(idx <= i){
        tbc = *(d_in + ((i+1-idx) * COLUMNS + (idx+1)));
        if(!tbc)
          *(d_out + ((i+1-idx) * COLUMNS + (idx+1))) = 0;
        else{
          left = *(d_out + ((i+1-idx) * COLUMNS + (idx)));
          up = *(d_out + ((i-idx) * COLUMNS + (idx+1)));
          corner = *(d_out + ((i-idx) * COLUMNS + (idx)));
          int mini = (left>up?up:left);
          *(d_out + ((i+1-idx) * COLUMNS + (idx+1))) = (mini>corner?corner:mini)+1;
        }
      }
    }
    __syncthreads();
  }

}

int S[ROWS][COLUMNS];

struct combine{
    int max;
    int *ptr;
};
struct combine ans_cpu;

void cpu_method(int *h_in_element){
    int i,j;      
    for(i = 0; i < ROWS; i++)  
        S[i][0] = *(h_in_element + i*COLUMNS);  
      
    for(j = 0; j < COLUMNS; j++)  
        S[0][j] = *(h_in_element + j); 
          
    for(i = 1; i < ROWS; i++){  
        for(j = 1; j < COLUMNS; j++){  
            if(*(h_in_element + (i*COLUMNS+j)) == 1){
                int left = S[i][j-1];
                int up = S[i-1][j];
                int corner = S[i-1][j-1];
                int mini = (left>up?up:left);  
                S[i][j] = (mini>corner?corner:mini) + 1;
              }  
            else
                S[i][j] = 0;  
        }  
    }
    int max_of_s = S[0][0], max_i, max_j;  
    for(i = 0; i < ROWS; i++){  
        for(j = 0; j < COLUMNS; j++){  
            if(max_of_s < S[i][j]){
                max_of_s = S[i][j];  
                max_i = i;  
                max_j = j;  
            }  
        }              
    }
    ans_cpu.max = max_of_s;
    ans_cpu.ptr = &S[0][0];  
}

int test_solution(int *ptr, int *h_out, int *h_in, int gpu_result, int cpu_result){
    int flag = 1;
    for(int i=0;i<ROWS;i++){
      int j;
      for(j=0;j<COLUMNS;j++){
        if(*(ptr + (i*COLUMNS+j)) != *(h_out + (i*COLUMNS+j))){
            //printf("%d, %d\n",i,j);
            flag = 0;
            break;
        }
      }
      if(j!=COLUMNS)
          break;
    }
    /*for(int i=0;i<ROWS;i++){
      for(int j=0;j<COLUMNS;j++){
        printf("%d ",*(h_in + (i*COLUMNS+j)));
      }
      printf("\n");
    }

    printf("\n");

    for(int i=0;i<ROWS;i++){
      for(int j=0;j<COLUMNS;j++){
        printf("%d ",*(ptr + (i*COLUMNS+j)));
      }
      printf("\n");
    }

    printf("\n");

    for(int i=0;i<ROWS;i++){
      for(int j=0;j<COLUMNS;j++){
        printf("%d ",*(h_out + (i*COLUMNS+j)));
      }
      printf("\n");
    }*/

    if(gpu_result != cpu_result){
        flag = 0;
    }
    return flag;
}

__global__ void find_max(int *d_final, int *d_max){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < ROWS*COLUMNS){
        int step = 1;
        while(step < blockDim.x){
            if(idx % (step*2) == 0 && (idx+step)<(ROWS*COLUMNS)){
                *(d_final + idx) = *(d_final + idx) > *(d_final + idx + step) ? *(d_final + idx) : *(d_final + idx + step);
            }
            __syncthreads();
            step *= 2;
        }
        if(threadIdx.x==0){
            atomicMax(d_max, *(d_final + idx));
        }
    }
}

int main(){

  srand(time(0));

  int h_in[ROWS*COLUMNS], h_out[ROWS*COLUMNS];

  for(int i=0;i<ROWS;i++){
    for(int j=0;j<COLUMNS;j++){
          *(h_in + (i*COLUMNS+j)) = rand() % 2;
    }
  }

  for(int i=0;i<COLUMNS;i++)
    *(h_out + (i)) = *(h_in + (i));

  for(int i=0;i<ROWS;i++)
    *(h_out + (i*COLUMNS)) = *(h_in + (i*COLUMNS));

  for(int i=1;i<ROWS;i++){
    for(int j=1;j<COLUMNS;j++){
      *(h_out + (i*COLUMNS+j)) = -1;
    }
  }

  int *d_in, *d_out, *d_final, *d_max;

  cudaMalloc(&d_in, MATRIX_BYTES);
  cudaMalloc(&d_out, MATRIX_BYTES);
  cudaMalloc(&d_final, MATRIX_BYTES);
  cudaMalloc(&d_max, sizeof(int));

  cudaMemcpy(d_in, h_in, MATRIX_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out, MATRIX_BYTES, cudaMemcpyHostToDevice);

  int min = ROWS>=COLUMNS?COLUMNS:ROWS;
  int THREADS;
  if(min>=1025)
    THREADS = 1024;
  else
    THREADS = min - 1;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int *h_max;
  cudaEventRecord(start);

  get_matrix<<<1,THREADS>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, MATRIX_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_final, h_out, MATRIX_BYTES, cudaMemcpyHostToDevice);
  find_max<<<CEIL((ROWS*COLUMNS), BLOCK_SIZE), BLOCK_SIZE>>>(d_final, d_max);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
  int gpu_result = *h_max;

  clock_t cpu_startTime, cpu_endTime;
  double cpu_ElapseTime=0;
  cpu_startTime = clock();

  cpu_method(&h_in[0]);
  int *ptr = ans_cpu.ptr;
  int cpu_result = ans_cpu.max;

  cpu_endTime = clock();
  cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/(1.0 * CLOCKS_PER_SEC)) * 1000;

  int flag = test_solution(ptr, &h_out[0], &h_in[0], gpu_result, cpu_result);

  if(flag) {
    printf("The computed matrix is correct!\n");
    printf("Time taken by GPU : %f ms\n", milliseconds);
    printf("Time taken by CPU : %f ms\n", cpu_ElapseTime);
    printf("The area of square is %d sq units\n",gpu_result*gpu_result);
  }
  else {
    printf("The computed matrix is incorrect!\n");
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_final);
  cudaFree(d_max);
  return 0;

}
