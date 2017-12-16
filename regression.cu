#include <stdio.h>
#define N 16
__device__ float sum(float *input){
	float sums=0;
	for(int i=0;i<N;i++){
		sums += input[i];
	}
	return sums;
}
__device__ float sum_of_power(float *input){
	float sums=0;
	for(int i=0;i<N;i++){
		sums += input[i]*input[i];
	}
	return sums;
}
__device__ float sum_of_mul(float *input1,float *input2){
	float sums=0;
	for(int i=0;i<N;i++){
		sums += input1[i]*input2[i];
	}
	return sums;
}
__device__ float mean(float *input1){
	float sums = sum(input1);
	float means = sums/N;
	return means;
}
__device__ float findSlope(float *input1,float *input2){
	float sumX,sumY,sum_mulXY,meanX,sum_powerX;
	sumX = sum(input1);
	sumY = sum(input2);
	sum_mulXY = sum_of_mul(input1,input2);
	sum_powerX = sum_of_power(input1);
	meanX = mean(input1);
	float b = float(sum_mulXY - float((sumX*sumY)/N))/(sum_powerX-float(N*(meanX*meanX)));
	return b;
}
__device__ float cut_point_Y(float *input1,float *input2,float b){
	float meanY = mean(input2);
	float meanX = mean(input1);
	float a = meanY - (meanX*b);
	return a;
}
__global__ void regression(float *input1,float *input2,float *out_a,float *out_b){
	// int tid = blockDim.x*blockIdx.x + threadIdx.x;
	*out_b = findSlope(input1,input2);
	*out_a = cut_point_Y(input1,input2,findSlope(input1,input2));
		// *output = input[tid];
		//  __syncthreads();
	// }
}
int main(){


	float x[] = {3.5,3,3.2,3.1,3.6,3.9,3.4,3.4,2.9,3.1,3.7,3.4,3,4,4.4,3.9,3.5,3.8,3.8,3.4,3.7,3.6,3.3,3.4,3,3.4,3.5,3.4,3.2,3.1,3.4,4.1,4.2,3.1,3.2,3.5,3.6,3,3.4,3.5,2.3,3.2,3.5,3.8,3,3.8,3.7};
	float y[] = {5.1,4.9,4.7,4.6,5,5.4,4.6,5,4.4,4.9,5.4,4.8,4.3,5.8,5.7,5.4,5.1,5.7,5.1,5.4,5.1,4.6,5.1,4.8,5,5,5.2,5.2,4.7,4.8,5.4,5.2,5.5,4.9,5,5.5,4.9,4.4,5.1,5,4.5,4.4,5,5.1,4.8,4.6,5.3};
	
	float b,a;
	float *d_x,*d_y,*out_a,*out_b;

	size_t size = N*sizeof(float);

	cudaMalloc(&d_x,size);
	cudaMalloc(&d_y,size);

	cudaMalloc(&out_a,sizeof(float));
	cudaMalloc(&out_b,sizeof(float));

	cudaMemcpy(d_x,x,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y,size,cudaMemcpyHostToDevice);

	regression<<<1,N>>>(d_x,d_y,out_a,out_b);

	cudaMemcpy(&a,out_a,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&b,out_b,sizeof(float),cudaMemcpyDeviceToHost);

	printf("%f \n", a+(b*3));

	cudaFree(d_x);
	cudaFree(out_a);
	cudaFree(out_b);
	return 0;
}