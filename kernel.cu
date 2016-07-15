
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <device_functions.h>

#define BlockNum 65535
#define ThreadNum 512

//=====================================================kmp算法预处理====================================================//
void getNext(char *pattern, int pattern_len, int *next)
{
	int len = 0;  // 记录前一个[最长匹配的前缀和后缀]的长度
	int i;
	next[0] = 0; // next[0] 必须是 0
	i = 1;
	// the loop calculates next[i] for i = 1 to pattern_len-1
	while (i < pattern_len)
	{
		if (pattern[i] == pattern[len])
		{
			len++;
			next[i] = len;
			i++;
		}
		else // (pat[i] != pat[len])
		{
			if (len == 0)
			{
				next[i] = len; //没有一个匹配的
				i++;
			}
			else // 如果 (len == 0)
			{
				// 这个地方有陷阱. 考虑这个例子 AAACAAAA ,i = 7.
				len = next[len - 1];
				// 另外, 注意 i 在这个地方并没有增加
			}
		}
	}
}
//======================================================kmp算法实现=====================================================//
__device__ void KMP(char *pattern, int pattern_len, char *array, int array_len, int *answer, int *next, int cursor, int end)
{
	//每个线程处理一个pattern_len数，即index的步长为id*pattern_len
	int j = 0;//j as index for pattern
	//cursor as index for array
	while (cursor < end)
	{
		if (pattern[j] == array[cursor])
		{
			j++;
			cursor++;
		}
		if (j == pattern_len)
		{
			//printf("Found pattern at index %d \n", i - j);
			answer[cursor - j] = 1;
			j = next[j - 1];
		}
		// mismatch after j matches
		else if (pattern[j] != array[cursor])
		{
			// Do not match next[0..next[j-1]] characters,
			// they will match anyway
			if (j != 0)
				j = next[j - 1];
			else
				cursor = cursor + 1;
		}
	}
}
//=====================================================kernel函数调用===================================================//
__global__ void kmp_kernel(char *arrayIn, char *patternIn, int *answerIn, int *next, int array_len, int pattern_len)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int offset= 2 * pattern_len; 
	int cursor, end;

		if (id < 0.5*(array_len / pattern_len))
		{
			cursor = id*offset;
			end = id*offset + offset;
		}
		else
		{ //辅助线程
			cursor = (id % ((array_len / pattern_len) / 2))*offset + offset - pattern_len;
			end = (id % ((array_len / pattern_len) / 2))*offset + offset + pattern_len;
		}

	KMP(patternIn, pattern_len, arrayIn, array_len, answerIn, next, cursor, end);
	//复制两个输入数组从 global memory 到 shared memory
	//__shared__ char array[blockDim.x+2*pattern_len];
}
//========================================================main函数======================================================//
int main()
{
	//=========================================================================//
	//error handling
	cudaError_t r;
	//host copies declaration
	char *array, *pattern; int *answer;
	//device copies declaration
	char *d_array, *d_pattern; int *d_answer;

	//=========================================================================//
	//input file operations & host arrays
	FILE * infile = fopen("input.txt", "r");
	if (infile == NULL){
		printf("ERROR:Could not open file '%s'.\n", "infile");
		exit(-1);
	}
	char readTemp;
	int array_len = 0; int pattern_len = 0;
	while ((readTemp = fgetc(infile)) != '\n') array_len++;
	while ((readTemp = fgetc(infile)) != '\n') pattern_len++;

	if (pattern_len > array_len || pattern_len < 0 || array_len < 0) { printf("ERROR INPUT!"); return 0; }
	bool zero_flag = false;
	if (pattern_len == 0 && array_len == 0) zero_flag = true;

	fseek(infile, 0, SEEK_SET);
	array = (char*)malloc(array_len*sizeof(char));
	pattern = (char*)malloc(pattern_len*sizeof(char));
	answer = (int*)malloc(array_len*sizeof(int));
	int readTemp1 = 0;
	while ((readTemp = fgetc(infile)) != '\n') { array[readTemp1] = readTemp; readTemp1++; }
	readTemp1 = 0;
	while ((readTemp = fgetc(infile)) != '\n') { pattern[readTemp1] = readTemp; readTemp1++; }
	fclose(infile);
	for (readTemp1 = 0; readTemp1 < array_len; readTemp1++) answer[readTemp1] = 0;

	//=========================================================================//
	//device arrays allocation
	r = cudaMalloc((void**)&d_array, sizeof(char)*array_len);
	printf("cudaMalloc d_array : %s\n", cudaGetErrorString(r));
	r = cudaMalloc((void**)&d_pattern, sizeof(char)*pattern_len);
	printf("cudaMalloc d_pattern : %s\n", cudaGetErrorString(r));
	r = cudaMalloc((void**)&d_answer, sizeof(int)*array_len);
	printf("cudaMalloc d_answer : %s\n", cudaGetErrorString(r));

	int* r_next = (int*)malloc(pattern_len*sizeof(int));
	//device
	int* next;
	r = cudaMalloc((void**)&next, sizeof(int)*pattern_len);
	printf("cudaMalloc next : %s\n", cudaGetErrorString(r));
	//preprocessing
	getNext(pattern, pattern_len, r_next);

	//=========================================================================//
	//cudaMemcpy for parameters
	r = cudaMemcpy(d_array, array, sizeof(char)*array_len, cudaMemcpyHostToDevice);
	printf("Memory copy H->D d_array : %s\n", cudaGetErrorString(r));
	r = cudaMemcpy(d_pattern, pattern, sizeof(char)*pattern_len, cudaMemcpyHostToDevice);
	printf("Memory copy H->D d_pattern : %s\n", cudaGetErrorString(r));
	//copy for next
	r = cudaMemcpy(next, r_next, sizeof(int)*pattern_len, cudaMemcpyHostToDevice);
	printf("Memory copy H->D d_pattern : %s\n", cudaGetErrorString(r));

	//=========================================================================//
		int threads = (array_len / pattern_len);//每个线程处理一个pattern长度的字符串
		int blocks = (threads / 1024) + 1;
		//call kernel
		kmp_kernel << <blocks, 2014 >> >(d_array, d_pattern, d_answer, next, array_len, pattern_len);

	r = cudaDeviceSynchronize();
	printf("Device synchronize : %s\n", cudaGetErrorString(r));
	//=========================================================================//
	//cudaMemcpy for result
	r = cudaMemcpy(answer, d_answer, sizeof(int)*array_len, cudaMemcpyDeviceToHost);
	printf("Memory copy D->H answer : %s\n", cudaGetErrorString(r));

	//test
	//int test;
	//for (test = 0; test < array_len; test++) printf("pos[%d]=%d\n", test, answer[test]);

	//=========================================================================//
	//output file operations
	FILE * outfile = fopen("output.txt", "w+");
	if (outfile == NULL){
		printf("ERROR:Could not open file '%s'.\n", "outfile");
		exit(-1);
	}
	if (zero_flag == false)
	{
		int writeTemp;
		bool flag = 0;
		for (writeTemp = 0; writeTemp < array_len; writeTemp++)
		if (answer[writeTemp] == 1)
		{
			if (flag == 0) flag = 1;
			fprintf(outfile, "%d\n", writeTemp);
		}
		if (flag == 0) fprintf(outfile, "Not found.");
	}
	else fprintf(outfile,"Null input.");
	fclose(outfile);

	//=========================================================================//
	//pointers free (host&device)
	free(array); free(pattern); free(answer); free(r_next);
	cudaFree(d_array); cudaFree(d_pattern); cudaFree(d_answer); cudaFree(next);

	return 1;
}
//======================================================================================================================//