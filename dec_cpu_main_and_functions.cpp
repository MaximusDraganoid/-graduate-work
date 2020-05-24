// ConsoleApplication1.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <iostream>

const int SIZE = 4;

template <int size> void standartMulKernelKar(unsigned int* a, unsigned int* b, unsigned int* c);
template <int size> void karatsubaMulOneStep(unsigned int *a, unsigned int *b, unsigned int *result);

int main()
{
	unsigned int a[] = {6, 7, 5, 3};
	unsigned int b[] = {9, 6, 2, 3};
	unsigned int c[SIZE * 2];
	for (int i = 0; i < SIZE * 2; i++) {
		c[i] = 0;
	}

	karatsubaMulOneStep <SIZE>(a, b, c);
	standartMulKernelKar <SIZE>(a, b, c);
	for (int i = 0; i < SIZE * 2; i++) {
		std::cout << c[i] << " ";
	}
	
	std::cout  << std::endl;

    return 0;
}

template <int size> void standartMulKernelKar(unsigned int* a, unsigned int* b, unsigned int* c) {

	unsigned long long int tempResult = 0;
	unsigned int carryArr[2 * size];
	unsigned int res[2 * size];

	for (int i = 0; i < 2 * size; i++) {
		carryArr[i] = 0;
	}

	for (int i = 0; i < 2 * size; i++) {
		res[i] = 0;
	}

	//само умножение
	for (int i = 0; i< size; i++) {
		for (int j = 0; j < size; j++) {
			tempResult = res[i + j] + a[i] * b[j];
			res[i + j] = tempResult % 10;
			carryArr[i + j + 1] += tempResult / 10;
		}
	}
	//распределение переноса
	for (int i = 0; i < 2 * size; i++) {
		tempResult = carryArr[i] + res[i];
		c[i] = tempResult % 10;
		if (i != 2 * size - 1) {
			carryArr[i + 1] += tempResult / 10;
		}
	}
	return;
}

template <int size> void karatsubaMulOneStep(unsigned int *a, unsigned int *b, unsigned int *result) {

	unsigned int *leftA = a;
	unsigned int *rightA = a + size / 2;

	unsigned int *leftB = b;
	unsigned int *rightB = b + size / 2;

	unsigned int middleResOne[size], middleResTwo[size], middleResThree[size], carryAdd[size];
	unsigned int res[2 * size];

	for (int i = 0; i < 2 * size; i++) {
		res[i] = 0;
	}

	for (int i = 0; i < size; i++) {
		middleResOne[i] = 0;
	}

	for (int i = 0; i < size; i++) {
		middleResTwo[i] = 0;
	}

	for (int i = 0; i < size; i++) {
		middleResThree[i] = 0;
	}

	for (int i = 0; i < size; i++) {
		carryAdd[i] = 0;
	}

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
		res[i + size / 2] = tempRes % 10;
		if (i != size - 1) {
			carryAdd[i + 1] += tempRes / 10;
		}
		else {
			perenos += tempRes / 10;
		}

	}

	for (int i = 0; i < size; i++) {
		tempRes = res[i + size / 2] + middleResTwo[i];
		res[i + size / 2] = tempRes % 10;
		if (i != size - 1) {
			carryAdd[i + 1] += tempRes / 10;
		}
		else {
			perenos += tempRes / 10;
		}
	}

	//carryDistribution
	for (int i = 0; i < size; i++) {
		tempRes = res[size / 2 + i] + carryAdd[i];
		res[size / 2 + i] = tempRes % 10;
		if (i != size - 1) {
			carryAdd[i + 1] += tempRes / 10;
		}
		else {
			perenos += tempRes / 10;
		}
	}

	for (int i = 0; i < size / 2; i++) {
		tempRes = res[size + size / 2 + i] + perenos;
		res[size + size / 2 + i] = tempRes % 10;
		perenos = tempRes / 10;
	}

	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

	int tempDiff = 0;
	//классическое вычитание 
	int zaem = 0;
	int base = 10;

	for (int i = 0; i < size / 2; i++) {
		tempDiff = int(rightA[i]);
		tempDiff -= zaem;
		tempDiff -= leftA[i];
		zaem = 0;
		if (tempDiff < 0) {
			tempDiff += 10;
			zaem = 1;
		}
		middleResOne[i] = tempDiff;
		if ((i == size / 2 - 1) && (zaem == 1)) {
			t_a = 1;
		}
	}

	int mask_a = 0;
	if (t_a == 1) {
		mask_a = 9;
		for (int i = 0; i < size / 2; i++) {
			middleResOne[i] = mask_a - middleResOne[i];
		}
		perenos = 1;
		for (int i = 0; i < size / 2; i++) {
			tempRes = middleResOne[i] + perenos;
			middleResOne[i] = tempRes % 10;
			perenos = tempRes / 10;
		}
	}

	//классическое вычитание
	zaem = 0;
	for (int i = 0; i < size / 2; i++) {
		//middleResTwo[i] = (rightB[i] - leftB[i] - zaem) % 10;
		tempDiff = int(rightB[i]);
		tempDiff -= zaem;
		tempDiff -= leftB[i];
		zaem = 0;
		if (tempDiff < 0) {
			tempDiff += 10;
			zaem = 1;
		}
		middleResTwo[i] = tempDiff;
		if ((i == size / 2 - 1) && (zaem == 1)) {
			t_b = 1;
		}
	}

	int mask_b = 0;
	if (t_b == 1) {
		mask_b = 9;
		for (int i = 0; i < size / 2; i++) {
			middleResTwo[i] = mask_b - middleResTwo[i];
		}
		perenos = 1;
		for (int i = 0; i < size / 2; i++) {
			tempRes = middleResTwo[i] + perenos;
			middleResTwo[i] = tempRes % 10;
			perenos = tempRes / 10;
		}
	}


	//вычисление произведений модулей
	standartMulKernelKar <size / 2>(middleResOne, middleResTwo, middleResThree);

	//вычитание произведения из центральных бит
	for (int i = 0; i < size; i++) {
		carryAdd[i] = 0;
	}

	//t_a ^ t_b
	//(t_a == t_b == 1) || (t_a == 1 && t_b == 0) || (t_a == 0 && t_b == 1)
	if (!(t_a ^ t_b)) {
		zaem = 0;
		for (int i = 0; i < size + size / 2; i++) {
			//middleResTwo[i] = (rightB[i] - leftB[i] - zaem) % 10;
			tempDiff = int(res[i + size / 2]);
			if (i < size) {
				tempDiff -= middleResThree[i];
			}
			tempDiff -= zaem;
			zaem = 0;
			if (tempDiff < 0) {
				tempDiff += 10;
				zaem = 1;
			}
			res[i + size / 2] = tempDiff;
		}

	}
	else {
		for (int i = 0; i < size; i++) {
			tempRes = res[i + size / 2] + middleResThree[i];
			res[i + size / 2] = tempRes % 10;
			carryAdd[i + 1] += tempRes / 10;
		}

		perenos = 0;
		for (int i = 0; i < size; i++) {
			tempRes = res[size / 2 + i] + carryAdd[i];
			res[size / 2 + i] = tempRes % 10;
			if (i != size - 1) {
				carryAdd[i + 1] += tempRes / 10;
			}
			else {
				perenos += tempRes / 10;
			}
		}

		for (int i = 0; i < size / 2; i++) {
			tempRes = res[size + size / 2 + i] + perenos;
			res[size + size / 2 + i] = tempRes % 10;
			perenos = tempRes / 10;
		}
	}

	for (int i = 0; i < 2 * size; i++) {
		result[i] = res[i];
	}

	return;
}
