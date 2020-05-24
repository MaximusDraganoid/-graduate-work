#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "stdlib.h"

const int SIZE = 8;
const int DOUBLE_SIZE = 16;
unsigned int const BASE = 65535;//10;//
const int KARATSUBA128 = 8;
const int MULTIPLIER = 1000;

__global__ void standartMulKernel(unsigned short int* a, unsigned short int* b, unsigned short int* c, unsigned short int* carryArr, int vectorsLen) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//calculations
	if (idx < vectorsLen) {
		int m = 0;// rows number
		int n = idx; // colomn number
		int temp = 0;

		while (n >= 0 && m <= vectorsLen - 1) {
			temp = temp + b[n] * a[m];
			m++;
			n--;
		}

		carryArr[idx + 1] = temp >> 16;
		c[idx] = (unsigned short int)temp;
	}
	else if (idx > vectorsLen - 1) {
		int n = SIZE - 1;
		int m = idx - n;
		int temp = 0;

		while (m <= vectorsLen - 1 && n >= 0) {
			temp = temp + b[n] * a[m];
			m++;
			n--;
		}

		carryArr[idx + 1] = temp >> 16;
		c[idx] = (unsigned short int)temp;
	}

	return;
}

template <int size> __device__ void standartMulKernelKar(unsigned int* a, unsigned int* b, unsigned int* c) {

	unsigned long long int tempResult = 0;
	unsigned int carryArr[2 * size];
	unsigned int res[2 * size];
	//само умножение
	for (int i = 0; i< size; i++) {
		for (int j = 0; j < size; j++) {
			tempResult = res[i + j] + a[i] * b[j];
			res[i + j] = (unsigned int)tempResult; //tempResult % 10;
			carryArr[i + j + 1] += tempResult >> 32; //tempResult / 10;
		}
	}
	//распределение переноса
	for (int i = 0; i < 2 * size; i++) {
		tempResult = carryArr[i] + res[i];
		c[i] = (unsigned int)tempResult; //tempResult % 10;
		if (i != 2 * size - 1) {
			carryArr[i + 1] += tempResult >> 32; //tempResult / 10;
		}
	}
	return;
}

__global__ void func_1(unsigned int* a, unsigned int* b, unsigned int* c) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = idx; i < MULTIPLIER; i = i + blockDim.x) {
			standartMulKernelKar<KARATSUBA128>(a + i * SIZE, b + i * SIZE, c + i * DOUBLE_SIZE);
	}
}

template <int size> __device__ void karatsubaMulOneStep(unsigned int *a, unsigned int *b, unsigned int *result) {

	unsigned int *leftA = a;
	unsigned int *rightA = a + size / 2;

	unsigned int *leftB = b;
	unsigned int *rightB = b + size / 2;

	unsigned int middleResOne[size], middleResTwo[size], middleResThree[size], carryAdd[size];
	unsigned int res[2 * size];

	unsigned long long int tempRes = 0;
	//А0*В0
	standartMulKernelKar <size / 2>(leftA, leftB, middleResOne);

	//А1*В1
	standartMulKernelKar <size / 2>(rightA, rightB, middleResTwo);

	//заполняем центральные значения 
	for (int i = 0; i < size; i++) {
		res[i] = middleResOne[i];
	}

	for (int i = 0; i < size; i++) {
		res[i + size] = middleResTwo[i];
	}

	int perenos = 0;
	//добавляем A0*B0 и A1*B1 к центральным байтам
	for (int i = 0; i < size; i++) {
		tempRes = res[i + size / 2] + middleResOne[i];
		res[i + size / 2] = (unsigned int)tempRes;
		if (i != size - 1) {
			carryAdd[i + 1] += tempRes >> 32;
		}
		else {
			perenos += tempRes >> 32;
		}

	}

	for (int i = 0; i < size; i++) {
		tempRes = res[i + size / 2] + middleResTwo[i];
		res[i + size / 2] = (unsigned int)tempRes;
		if (i != size - 1) {
			carryAdd[i + 1] += tempRes >> 32;
		}
		else {
			perenos += tempRes >> 32;
		}
	}

	//carryDistribution
	for (int i = 0; i < size; i++) {
		tempRes = res[size / 2 + i] + carryAdd[i];
		res[size / 2 + i] = (unsigned int)tempRes;
		if (i != size - 1) {
			carryAdd[i + 1] += tempRes >> 32;
		}
		else {
			perenos += tempRes >> 32;
		}
	}

	for (int i = 0; i < size / 2; i++) {
		tempRes = res[size + size / 2 + i] + perenos;
		res[size + size / 2 + i] = (unsigned int)tempRes;
		perenos = tempRes >> 32;
	}

	for (int i = 0; i < size; i++) {
		carryAdd[i] = 0;
	}

	//вычисляем разности
	unsigned short int t_a = 0;
	unsigned short int t_b = 0;

	for (int i = 0; i < size; i++) {
		middleResOne[i] = 0;
	}

	for (int i = 0; i < size; i++) {
		middleResTwo[i] = 0;
	}

	unsigned int tempDiff = 0;
	//классическое вычитание 
	unsigned int zaem = 0;
	int base = 10;

	for (int i = 0; i < size / 2; i++) {
		middleResOne[i] = rightA[i] - leftA[i] - zaem;
		
		zaem = 0;
		if (middleResOne[i] > rightA[i]) {
			zaem = 1;
		}

		if ((i = size / 2 - 1) && (zaem == 1)) {
			t_a = 1;
		}
	}
	//вычисляем модуль разности в случае 
	if (t_a == 1) {
		unsigned int mask_a = -t_a;
		for (int i = 0; i < size / 2; i++) {
			middleResOne[i] = middleResOne[i] ^ mask_a;
		}
		perenos = t_a;
		for (int i = 0; i < size / 2; i++) {
			tempRes = middleResOne[i] + perenos;
			middleResOne[i] = (unsigned int)tempRes;
			perenos = tempRes >> 32;
		}
	}
	
	//классическое вычитание
	zaem = 0;
	for (int i = 0; i < size / 2; i++) {
		middleResTwo[i] = rightB[i] - leftB[i] - zaem;
		zaem = 0;
		if (middleResTwo[i] > rightB[i]) {
			zaem = 1;
		}

		if ((i = size / 2 - 1) && (zaem == 1)) {
			t_b = 1;
		}
	}

	if (t_b == 1) {
		unsigned int mask_b = -t_b;
		for (int i = 0; i < size / 2; i++) {
			middleResTwo[i] = middleResTwo[i] ^ mask_b;
		}
		perenos = t_b;
		for (int i = 0; i < size / 2; i++) {
			tempRes = middleResTwo[i] + perenos;
			middleResTwo[i] = (unsigned int)tempRes;
			perenos = tempRes >> 32;
		}
	}

	//вычисление произведений модулей
	standartMulKernelKar <size / 2>(middleResOne, middleResTwo, middleResThree);
	//вычитание произведения из центральных бит
	for (int i = 0; i < size; i++) {
		carryAdd[i] = 0;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	if (!(t_a ^ t_b)) {
		zaem = 0;
		for (int i = 0; i < size; i++) {
			tempDiff = res[i + size / 2] - zaem - middleResThree[i];
			zaem = 0;
			if (tempDiff > res[i + size / 2]) {
				zaem = 1;
			}
			res[i + size / 2] = tempDiff;
		}

		for (int i = 0; i < size / 2; i++) {
			tempDiff = res[i + size + size / 2] - zaem;
			zaem = 0;
			if (tempDiff > res[i + size + size / 2]) {
				zaem = 1;
			}
			res[i + size + size / 2] = tempDiff;
		}
	} else {
		for (int i = 0; i < size; i++) {
			tempRes = res[i + size / 2] + middleResThree[i];
			res[i + size / 2] = (unsigned int)tempRes;
			if (i != size - 1) {
				carryAdd[i + 1] += tempRes >> 32;
			}
			else {
				perenos += tempRes >> 32;
			}
		}

		perenos = 0;
		for (int i = 0; i < size; i++) {
			tempRes = res[size / 2 + i] + carryAdd[i];
			res[size / 2 + i] = (unsigned int)tempRes;
			if (i != size - 1) {
				carryAdd[i + 1] += tempRes >> 32;
			}
			else {
				perenos += tempRes >> 32;
			}
		}

		for (int i = 0; i < size / 2; i++) {
			tempRes = res[size + size / 2 + i] + perenos;
			res[size + size / 2 + i] = (unsigned int)tempRes;
			perenos = tempRes >> 32;
		}
	}

	for (int i = 0; i < 2 * size; i++) {
		result[i] = res[i];
	}

	return;
}


__global__ void func_2(unsigned int *a, unsigned int *b, unsigned int *result) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = idx; i < MULTIPLIER; i = i + blockDim.x) {
		karatsubaMulOneStep<KARATSUBA128>(a + i * SIZE, b + i * SIZE, result + i * DOUBLE_SIZE);
	}
}


void main()
{
	//std::cout << "size of long int is  " << sizeof(long long int) << std::endl;
	cudaError_t cudaStatus;

	unsigned int a[SIZE * MULTIPLIER];
	unsigned int b[SIZE * MULTIPLIER];
	unsigned int resArrray[DOUBLE_SIZE * MULTIPLIER];
	for (int i = 0; i < SIZE * MULTIPLIER; i++) {
		a[i] = rand();
		b[i] = rand();
	}

	unsigned int* a_device;
	unsigned int* b_device;
	unsigned int* resArrray_device;
	unsigned int* resCarryArr;

	cudaStatus = cudaMalloc((void**)&a_device, sizeof(unsigned int) * SIZE * MULTIPLIER);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 1");
		return;
	}

	cudaStatus = cudaMalloc((void**)&b_device, sizeof(unsigned int) * SIZE * MULTIPLIER);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 2");
		return;
	}

	cudaStatus = cudaMalloc((void**)&resArrray_device, sizeof(unsigned int) * DOUBLE_SIZE * MULTIPLIER);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 3");
		return;
	}



	cudaStatus = cudaMemcpy(a_device, a, sizeof(unsigned int) * SIZE * MULTIPLIER, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 4 ");
		return;
	}

	cudaStatus = cudaMemcpy(b_device, b, sizeof(unsigned int) * SIZE * MULTIPLIER, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 5");
		return;
	}
	//замеры времени
	cudaEvent_t start, stop;
	
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);//начало отсчета

	func_2 <<<1, 1>>> (a_device, b_device, resArrray_device);

	//func_1 <<<1, 1>>> (a_device, b_device, resArrray_device);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	
	std::cout << cudaGetLastError() << std::endl;

	// print the cpu and gpu times
	printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);

	// release resources
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 6 ");
		return;
	}

	cudaStatus = cudaMemcpy(resArrray, resArrray_device, sizeof(unsigned int) * DOUBLE_SIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 7");
			std::cout << std::endl  << cudaStatus << std::endl;
		//return;
	}


    return;
}

