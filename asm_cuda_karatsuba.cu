#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "stdlib.h"

const int SIZE = 8;
const int DOUBLE_SIZE = 16;
unsigned int const BASE = 65535;//10;//
const int KARATSUBA128 = 8;
const int MULTIPLIER = 1000;

template <int size> __device__ void standartMulKernelKar(unsigned int* a, unsigned int* b, unsigned int* c) {

	for (int i = 0; i < size; i++) {
		c[i] = a[0] * b[i];
		c[i + size] = 0;
	}

	asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;\n\t" :\
		"=r"(c[1]) : "r"(a[0]), "r"(b[0]), "r"(c[1]));
	for (int i = 1; i < size - 1; i++) {
		asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t" :\
			"=r"(c[i + 1]) : "r"(a[0]), "r"(b[i]), "r"(c[i + 1]));
	}
	asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t" :\
		"=r"(c[size]) : "r"(a[0]), "r"(b[size - 1]), "r"(c[size]));
	for (int i = 1; i < size; i++) {
		unsigned int t = a[i];
		asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;\n\t" :\
			"=r"(c[i]) : "r"(t), "r"(b[0]), "r"(c[i]));
		for (int j = 1; j < size; j++) {
			asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;\n\t" :\
				"=r"(c[i + j]) : "r"(t), "r"(b[j]), "r"(c[i + j]));
		}
		asm volatile("addc.u32 %0, %1, %2;\n\t" :\
			"=r"(c[i + size]) : "r"(c[i + size]), "r"(0));
		asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;\n\t" :\
			"=r"(c[i + 1]) : "r"(t), "r"(b[0]), "r"(c[i + 1]));
		for (int j = 1; j < size - 1; j++) {
			asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t" :\
				"=r"(c[j + i + 1]) : "r"(t), "r"(b[j]), "r"(c[i + j + 1]));
		}
		asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t" :\
			"=r"(c[i + size]) : "r"(t), "r"(b[size - 1]), "r"(c[i + size]));
	}
}

__global__ void func_1(unsigned int* a, unsigned int* b, unsigned int* c) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = idx; i < MULTIPLIER; i = i + blockDim.x) {
		standartMulKernelKar<KARATSUBA128>(a + i * SIZE, b + i * SIZE, c + i * DOUBLE_SIZE);
	}
}


template <int size> __device__ void karatsubaMul_128(unsigned int *a, unsigned int *b, unsigned int *result) {

	unsigned int *leftA = a;
	unsigned int *rightA = a + 4;

	unsigned int *leftB = b;
	unsigned int *rightB = b + 4;

	unsigned int middleResOne[size], middleResTwo[size], middleResThree[size];
	unsigned int res[2 * size];
	unsigned int tempRes = 0;
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

	//добавляем A0*B0 и A1*B1 к центральным байтам
	//+a0*b0
	asm volatile("add.cc.u32 %0, %1, %2;\n\t":\
		"=r"(res[size / 2]) : "r"(res[size / 2]), "r"(middleResOne[0]));
	
	for (int i = 1; i < size; i++) {
		asm volatile("addc.cc.u32 %0, %1, %2;\n\t":\
			"=r"(res[i + size / 2]):"r"(res[i + size / 2]), "r"(middleResOne[i]));
	}
	//вот здесь возникает вопрос: все ли будет в порядке с флагом cc.cf? 
	//распределение переноса
	for (int i = 0; i < size / 2; i++) {
		asm volatile("addc.cc.u32 %0, %1, 0;\n\t":\
			"=r"(res[i + size / 2 + size]) : "r"(res[i + size / 2 + size]));
	}

	//+a1*b1
	asm volatile("add.cc.u32 %0, %1, %2;\n\t":\
		"=r"(res[size / 2]) : "r"(res[size / 2]), "r"(middleResTwo[0]));

	for (int i = 1; i < size; i++) {
		asm volatile("addc.cc.u32 %0, %1, %2;\n\t":\
			"=r"(res[i + size / 2]) : "r"(res[i + size / 2]), "r"(middleResTwo[i]));
	}
	//вот здесь возникает вопрос: все ли будет в порядке с флагом cc.cf? 
	//распределение переноса
	for (int i = 0; i < size / 2; i++) {
		asm volatile("addc.cc.u32 %0, %1, 0;\n\t":\
			"=r"(res[i + size / 2 + size]) : "r"(res[i + size / 2 + size]));
	}

	//вычисляем разности
	unsigned int t_a = 0;
	unsigned int t_b = 0;

	for (int i = 0; i < size; i++) {
		middleResOne[i] = 0;
	}

	for (int i = 0; i < size; i++) {
		middleResTwo[i] = 0;
	}

	//классическое вычитание 
	asm volatile("sub.cc.u32 %0, %1, %2;\n\t":\
		"=r"(middleResOne[0]) : "r"(rightA[0]), "r"(leftA[0]));
	for (int i = 1; i < size / 2; i++) {
		asm volatile("subc.cc.u32 %0, %1, %2;\n\t":\
			"=r"(middleResOne[i]) : "r"(rightA[i]), "r"(leftA[i]));
	}
	//запоминаем бит заема
	asm volatile("addc.u32 %0, 0, 0;\n\t" : "=r"(t_a) : );

	if (t_a) {
		unsigned int mask_a = -t_a;
		for (int i = 0; i < size / 2; i++) {
			middleResOne[i] = middleResOne[i] ^ mask_a;
		}
		asm volatile("add.cc.u32 %0, %1, %2;\n\t":\
			"=r"(middleResOne[0]) : "r"(middleResOne[0]), "r"(t_a));
		for (int i = 0; i < size / 2; i++) {
			asm volatile("addc.cc.u32 %0, %1, %2;\n\t":\
				"=r"(middleResOne[i]) : "r"(middleResOne[i]), "r"(0));
		}
	}

	//классическое вычитани
	asm volatile("sub.cc.u32 %0, %1, %2;\n\t":\
		"=r"(middleResTwo[0]) : "r"(rightB[0]), "r"(leftB[0]));
	for (int i = 1; i < size / 2; i++) {
		asm volatile("subc.cc.u32 %0, %1, %2;\n\t":\
			"=r"(middleResTwo[i]) : "r"(rightB[i]), "r"(leftB[i]));
	}
	//запоминаем бит заема
	asm volatile("addc.u32 %0, 0, 0;\n\t" : "=r"(t_b) : );

	if (t_b == 1) {
		unsigned int mask_b = -t_b;
		for (int i = 0; i < size / 2; i++) {
			middleResTwo[i] = middleResTwo[i] ^ mask_b;
		}
		asm volatile("add.cc.u32 %0, %1, %2;\n\t":\
			"=r"(middleResTwo[0]) : "r"(middleResTwo[0]), "r"(t_b));
		for (int i = 0; i < size / 2; i++) {
			asm volatile("addc.cc.u32 %0, %1, %2;\n\t":\
				"=r"(middleResTwo[i]) : "r"(middleResTwo[i]), "r"(0));
		}
	}

	//вычисление произведений модулей
	standartMulKernelKar <size / 2>(middleResOne, middleResTwo, middleResThree);

	if (!(t_a ^ t_b)) {
		asm volatile("sub.cc.u32 %0, %1, %2;\n\t":\
			"=r"(res[size / 2]) : "r"(rightB[size / 2]), "r"(middleResThree[0]));

		for (int i = 1; i < size; i++) {
			asm volatile("subc.cc.u32 %0, %1, %2;\n\t":\
				"=r"(res[i + size / 2]) : "r"(res[i + size / 2]), "r"(middleResThree[i]));
		}
		//распределение заема
		for (int i = 0; i < size / 2; i++) {
			asm volatile("subc.cc.u32 %0, 0, 0;\n\t":\
				"=r"(res[i + size / 2 + size]) : );
		}
	}
	else {
		asm volatile("add.cc.u32 %0, %1, %2;\n\t":\
			"=r"(res[size / 2]) : "r"(res[size / 2]), "r"(middleResThree[0]));

		for (int i = 1; i < size; i++) {
			asm volatile("addc.cc.u32 %0, %1, %2;\n\t":\
				"=r"(res[i + size / 2]) : "r"(res[i + size / 2]), "r"(middleResThree[i]));
		}
		//распределение переноса
		for (int i = 0; i < size / 2; i++) {
			asm volatile("addc.cc.u32 %0, %1, 0;\n\t":\
				"=r"(res[i + size / 2 + size]) : "r"(res[i + size / 2 + size]));
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
		karatsubaMul_128<KARATSUBA128>(a + i * SIZE, b + i * SIZE, result + i * DOUBLE_SIZE);
	}
}


void main()
{
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

	//func_1 <<<1, 16>>> (a_device, b_device, resArrray_device);

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
		std::cout << std::endl << cudaStatus << std::endl;
		//return;
	}


	return;
}

