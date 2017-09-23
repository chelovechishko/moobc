#include <thread>
#include <Windows.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <list>
#include <stdlib.h>
#include <math.h>
#include <string>

using namespace std;
#define DIMENSIONS 10
#define ARC 13
typedef float point[ DIMENSIONS ];
CRITICAL_SECTION Locker;
struct Device   //краткая информацию о девайсах для рабочих потоков
{
	int SystemDeviceId; //идентификатор видеокарты для запуска
	int WarpSize;
	int MaxElemComputeCapability; //число элементов(точек 10ми мерного пространства), которое видеокарта может обрабатывать одновременно
	Device(int systemDeviceId,int WarpSize, int maxElemComputeCapability)
	{
		SystemDeviceId = systemDeviceId;
		this->WarpSize = WarpSize;
		MaxElemComputeCapability = maxElemComputeCapability;
	}
	Device(const Device& d)
	{
		SystemDeviceId = d.SystemDeviceId;
		this->WarpSize = d.WarpSize;
		MaxElemComputeCapability = d.MaxElemComputeCapability;
	}
};

struct Work  //вся инфомация для рабочих потоков для запуска кернела собрана тут
{
	int DeviceIdToPerform;  //идентификатор видеокарты для запуска
	dim3 BlockAm;
	dim3 ThreadPerBlock;
	bool ExtremaType;
	size_t Elements_Am;
	float* x_arr;
	Work()
	{}
	Work(const int& deviceIdToPerform, const size_t& elems_am, const dim3& blockAm, const dim3& threadPerBlock,  float* x_array, bool extremaType)
	{
		DeviceIdToPerform = deviceIdToPerform;
		BlockAm = blockAm;
		ThreadPerBlock = threadPerBlock;
		x_arr = x_array;
		Elements_Am = elems_am;
		ExtremaType = extremaType;
	}
};
size_t ElemsComputeCapabiity;  // общее число точек (10и мерных) которое все видеокарты могут обрабатывать одновременно
list<float> ChunckExtrema; // экстремумы которые были найдены в одном чанке
list<float> GlobalExtrema; // глобально найденные экстремумы
float GlobalIntStart[DIMENSIONS], GlobalIntEnd[DIMENSIONS];  //крайние точки каждого измерения в исследуемом файле
float ChoosedIntStart[DIMENSIONS], ChoosedIntEnd[DIMENSIONS]; //выбранная пользователем область поиска

  __host__ __device__ float Ft_10D(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7, float x8, float x9)
{
	
	return x1+x2+x3+x4+x5+x6+x7+x8+x9+x0;
}
  __device__ __host__ float F(point x)
 {

	 return x[0]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[1];

 }

__device__ void point_copy( point in, point out ) {
	if( in == out ) {
		return;
		}
	for( int i = 0; i < DIMENSIONS; ++i ) {
		out[ i ] = in[ i ];
		}
	}
__device__ bool isInSearchArea(point point_to_check, point intStart, point intEnd)
{
	for(int i = 0; i < DIMENSIONS; ++i)
	{
		if ((point_to_check[i] < intStart[i]) || (point_to_check[i] > intEnd[i])) return false;
	}
	return true;
}
__device__ void min_or_max( bool is_min, point a, point b)
{
		float res_a = F(a);
		float res_b = F(b);
		if(res_a != res_a && res_b != res_b) //if both nan do nothing
		{
			 return;
		}
		if(isinf(res_a) || isinf(res_b))
		{
			return;
		}
		if(isnan(res_a))
		{
			 point_copy( b, a );
			 return; }
		if( is_min )
		{
			if(res_b < res_a) point_copy( b, a );
		}
		else
		{
			if(res_b > res_a) point_copy(b, a);
		}
		return;
	
}
// compare_and_replace
__device__ void atomic_car( bool is_min, float* in, float* out, float* mutex) 
{
		bool min = false;
		bool max = false;
		float calculated_in = F( in );  //if f(in) == nan there will not be copy
		while( atomicCAS( ( int* )mutex, 0, 1 ) ) {
			// sleep
		}
		if (isfinite(calculated_in) && calculated_in == calculated_in) // calculated_in != nan
		{
			if (!isnan(out[0])) // if buffer is not empty
			{
				float calculated_out =  F( out );
				min = is_min && calculated_in < calculated_out;
				max = !is_min && calculated_in > calculated_out;
			}
			else
				min = true;
			/*
			if both f(in) and f(out) == nan there wont be any copy
			host code won't print answer if buffer contains only nan
			*/
		}
		if( min || max ) {
			point_copy( in, out );
		}
		atomicDec( ( unsigned int* )mutex, 1 ); 
	
}
__device__ void loadtoShared(point global_p, point cahce_p, point IntStart, point intEnd)
{
	if (isInSearchArea(global_p, IntStart, intEnd))
	{
		point_copy(global_p, cahce_p);  
	}
	else
	{
		for(int i = 0; i < DIMENSIONS; ++i)
		{
			cahce_p[i] = sqrtf(-1);
		}
	}
}

__global__ void findChunkExtrema(point d_x_arr[], float* d_extrema_buff, point intStart, point intEnd, int chunkSize, bool is_min)
{
	int thread = threadIdx.x;
	extern __shared__ point cache[];
	int global_d_x_arr_index = blockDim.x * blockIdx.x + thread;
	if (global_d_x_arr_index < chunkSize) // if element exists
	{
		loadtoShared(d_x_arr[global_d_x_arr_index], cache[thread], intStart, intEnd ); 
		__syncthreads();
		for( int step = blockDim.x / 2; step; step >>= 1 )
		{
			if( thread < step )
			{
				if(global_d_x_arr_index + step < chunkSize)  // if element to compare in this chunk
				{
					min_or_max(is_min, cache[thread], cache[thread + step]);
					
				}
			}
			__syncthreads();
		}
	}
	if( !thread )
	{
		atomic_car( is_min, (float*) cache, d_extrema_buff,( float* )d_extrema_buff - 1);
		/*Now all blocks has fixed amount of threads = warp size. Even if there are less then warpsize elements.
		it means that all steps can be divided by 2 without any remain. We should only check global index,
		it must be less than chunck size, that provided as function param.
		*/
	}
}


//сравнивает две последовательности экстремумов и записывает в InitialExtremaChunk меньшую/большую найденную последовательность 
void compareExtrema(list<float>& InitialExtremaChunk, list<float>& anotherChunckExtrema, bool ExtremaType)
{
	float AnotherRes = 0;
	float CurrentRes = 0;
	if (InitialExtremaChunk.size() !=0 && anotherChunckExtrema.size() != 0)
	{
		
			 list<float>::iterator it = anotherChunckExtrema.begin();
			 float t1 = *it;
			 float t2 = *(++it);
			 float t3 = *(++it);
			 float t4 = *(++it);
			 float t5 = *(++it);
			 float t6 = *(++it);
			 float t7 = *(++it);
			 float t8 = *(++it);
			 float t9 = *(++it);
			 float t10 = *(++it);
			 AnotherRes = Ft_10D(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10);
			 it = InitialExtremaChunk.begin();
			  t1 = *it;
			  t2 = *(++it);
			  t3 = *(++it);
			  t4 = *(++it);
			  t5 = *(++it);
			  t6 = *(++it);
			  t7 = *(++it);
			  t8 = *(++it);
			  t9 = *(++it);
			  t10 = *(++it);
			 CurrentRes = Ft_10D(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10);
	}
	if (ExtremaType && (AnotherRes <= CurrentRes) || (!ExtremaType && (AnotherRes >= CurrentRes))) 
			{
				if (ExtremaType && (AnotherRes < CurrentRes) || (!ExtremaType && (AnotherRes > CurrentRes))) //(если значение функции меньше/больше
				{
					InitialExtremaChunk.clear(); // чистим старый буфер)
				}
				InitialExtremaChunk.splice(InitialExtremaChunk.end(), anotherChunckExtrema); ///обьединяем последовательности
	} 
}
//перегрузка той же функции но для данных из кернела
void compareExtrema(list<float>& initialExtremaChunk, float*& anotherChunckExtrema, bool ExtremaType)
{
	float AnotherRes = 0;
	float CurrentRes = 0;
	if (initialExtremaChunk.size() !=0)
	{
		
				AnotherRes = Ft_10D(anotherChunckExtrema[0], anotherChunckExtrema[1], anotherChunckExtrema[2],anotherChunckExtrema[3],anotherChunckExtrema[4],anotherChunckExtrema[5],anotherChunckExtrema[6],anotherChunckExtrema[7],anotherChunckExtrema[8],anotherChunckExtrema[9]);
				list<float>::iterator it = initialExtremaChunk.begin();
				
			  float t1 = *it;
				 float t2 = *(++it);
				 float t3 = *(++it);
				 float t4 = *(++it);
				 float t5 = *(++it);
				 float t6 = *(++it);
				 float t7 = *(++it);
				 float t8 = *(++it);
				 float t9 = *(++it);
				 float t10 = *(++it);
			 CurrentRes = Ft_10D(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10);

	}
	if ((ExtremaType && (AnotherRes <= CurrentRes)) || (!ExtremaType &&  (AnotherRes >= CurrentRes)))
			{
				if (ExtremaType && (AnotherRes < CurrentRes) || (!ExtremaType && (AnotherRes > CurrentRes)))
				{
					initialExtremaChunk.clear(); //clear old
				} 
				for(int i = 0; i < DIMENSIONS && (!_isnanf(anotherChunckExtrema[i])); ++i)
				{
					int s = initialExtremaChunk.size();
					initialExtremaChunk.push_back(anotherChunckExtrema[i]);
				}
			}
}
//функция запускающая рассчет задачи на видеокарте
void compute(Work Task)
{
	cudaSetDevice(Task.DeviceIdToPerform);   //устанавливаем девайс для запуска
	int ex_memory_am = DIMENSIONS * sizeof(float) + sizeof(float);  //вычисляем память для буфера
	float* ExtremaBuffer = new float[ex_memory_am];      //создаем буфер 
	ExtremaBuffer[0] = 0;   //нулевая позиция используется в качестве мьютекса
	point* d_x_arr;
	float* d_extrema_buff; 
	float* d_choosed_int_start;
	float* d_choosed_int_end;
	for(int i = 0; i < DIMENSIONS; ++i)
	{
		ExtremaBuffer[i + 1] = sqrtf(-1);  //заполняем буфер NaN
	}
	cudaMalloc((void**)&d_x_arr,sizeof(float) * Task.Elements_Am * DIMENSIONS);
	cudaMalloc((void**)&d_extrema_buff, sizeof(float) * DIMENSIONS + sizeof(float));
	cudaMalloc((void**)&d_choosed_int_start,sizeof(float) * DIMENSIONS);
	cudaMalloc((void**)&d_choosed_int_end, sizeof(float) * DIMENSIONS);
	cudaMemcpy(d_x_arr, Task.x_arr, Task.Elements_Am * sizeof(float) * DIMENSIONS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_extrema_buff, ExtremaBuffer, ex_memory_am, cudaMemcpyHostToDevice);
	cudaMemcpy(d_choosed_int_start, ChoosedIntStart, DIMENSIONS * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_choosed_int_end, ChoosedIntEnd, DIMENSIONS * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	int CacheMemoryPerBlock = sizeof(float) * Task.ThreadPerBlock.x * DIMENSIONS; //вычисляем количество shared памяти

	findChunkExtrema<<<Task.BlockAm, Task.ThreadPerBlock, CacheMemoryPerBlock>>>(d_x_arr, d_extrema_buff + 1, d_choosed_int_start, d_choosed_int_end, Task.Elements_Am,  Task.ExtremaType);
	cudaMemcpy(ExtremaBuffer, d_extrema_buff + 1, ex_memory_am - sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	EnterCriticalSection(&Locker);
	if (ExtremaBuffer[0] == ExtremaBuffer[0]) //если были найдены экстремумы
	{
		compareExtrema(ChunckExtrema, ExtremaBuffer, Task.ExtremaType); //сравниваем их с найденными на текущей итерации
	}
	LeaveCriticalSection(&Locker);
	cudaFree(d_x_arr);
	cudaFree(d_extrema_buff);
	cudaFree(d_choosed_int_start);
	cudaFree(d_choosed_int_end);
}
//создает поток и запсукает его
void launch_GPU_computation(thread& Worker, Work Task)
{
	Worker = thread(compute, Task);
}
//приостанавливает поток родитель пока ребенок не закончит работу
void wait_GPU_results(thread& Worker)
{
	if (Worker.joinable())
	{
		Worker.join();
	}
}

//читает чанк данных из source, и находит на участке экстремум
void findLocalExtrema(ifstream& source, list<Device>& Devices, thread*& Workers, size_t& ElementsLeft, bool extrema_type)
{
	float* x_arr;
	float* x_arr1;
	ChunckExtrema.clear(); //очищаем буфер от предыдущих результатов
	if(ElementsLeft >= ElemsComputeCapabiity)  // если точек все еще много запускаем все видеокарты на полную мощность
	{
		x_arr = (float*)new float[ElemsComputeCapabiity * DIMENSIONS];
		x_arr1 = x_arr;
		source.read((char*)x_arr, sizeof(float) * ElemsComputeCapabiity * DIMENSIONS); 
		int thr_id = 0;
		for(list<Device>::iterator It = Devices.begin(); It != Devices.end(); ++It)  //по очереди запускаем каждую видеокарту
		{
			Work task((*It).SystemDeviceId, (*It).MaxElemComputeCapability, (*It).MaxElemComputeCapability / (*It).WarpSize ,(*It).WarpSize, x_arr, extrema_type);
			launch_GPU_computation(Workers[thr_id++], task);
			x_arr += (*It).MaxElemComputeCapability * DIMENSIONS;
			ElementsLeft -= (*It).MaxElemComputeCapability;
		}
	}
	else // иначе запускам оптимальное кол-во видеокарт
	{
		//читаем остаток файла
		x_arr = (float*)new char[sizeof(float) * ElementsLeft * DIMENSIONS];
		x_arr1 = x_arr;
		source.read((char*)x_arr, sizeof(float) * ElementsLeft * DIMENSIONS);
		size_t CurrentComputeCapability = ElemsComputeCapabiity;				
		//с конца массива девайсов, мы ищем лучшее кол-во видеокарт для запуска
		for(list<Device>::reverse_iterator It = Devices.rbegin(); It != Devices.rend(); ++It)
		{
			if(ElementsLeft > (CurrentComputeCapability -= (*It).MaxElemComputeCapability))
			{
				//здесь, после того как мы нашли ту видеокарту для которой не надо запсукать всю мощность, запускаем все предыдущие видеокарты, 
				//затем рассчитываем подоходящее число блоков для остатков
				int thr_id = 0;
				for(list<Device>::iterator It1 = Devices.begin(); (*It1).SystemDeviceId != (*It).SystemDeviceId; ++It1) 
				{
					Work task((*It1).SystemDeviceId, (*It1).MaxElemComputeCapability,  (*It).MaxElemComputeCapability / (*It).WarpSize , (*It).WarpSize, x_arr, extrema_type);
					launch_GPU_computation(Workers[thr_id++], task);
					x_arr += (*It1).MaxElemComputeCapability * DIMENSIONS;
					ElementsLeft -= (*It1).MaxElemComputeCapability;
				}
				// запускаем последний гпу
				Work task((*It).SystemDeviceId, ElementsLeft, ceil(ElementsLeft / (float)(*It).WarpSize), (*It).WarpSize , x_arr, extrema_type);
				launch_GPU_computation(Workers[thr_id], task);
				ElementsLeft -= ElementsLeft;
				break;
			}
		}
	}
	for(int i = 0; i < Devices.size(); ++i)
	{
		wait_GPU_results(Workers[i]);
	}
	free(x_arr1);
}

int main()
{
	setlocale(LOCALE_ALL, "rus");
	bool min;
	cout << "Введите тип экстремума который хотите найти (1 - min 0 - max) " << endl;
	cin >> min;
	int DeviceCount;
	cudaGetDeviceCount(&DeviceCount);
	if(DeviceCount > 0)
	{
		list<Device> Devices;
		cudaDeviceProp props;
		for(int i = 0; i < DeviceCount; ++i)
		{
			cudaGetDeviceProperties(&props, i);
			if ( props.major > ARC / 10 || ( props.major == ARC / 10 && props.minor >= ARC % 10 ) )//((props.major >= ARC / 10) || (props.minor >= ARC % 10))
			{
				int MaxConcurentThreadAm =  props.multiProcessorCount * props.maxThreadsPerMultiProcessor;
				int ElemsCC = MaxConcurentThreadAm;
				int AvailableSharedMem =  props.sharedMemPerMultiprocessor;
				int elemsize = MaxConcurentThreadAm * DIMENSIONS * sizeof(float);
				if(elemsize > AvailableSharedMem)
				{
					ElemsCC = AvailableSharedMem / ((sizeof(float) * DIMENSIONS)); //will divide with remain
				}
				Device WorkGpu(i, props.warpSize, ElemsCC);
				Devices.push_back(WorkGpu);
			}
		}
		if(Devices.size() > 0)
		{
		    InitializeCriticalSection(&Locker);
			thread* GpuWorkers = new thread[Devices.size()];
			for(list<Device>::iterator It = Devices.begin(); It != Devices.end(); ++It)
			{
				ElemsComputeCapabiity += (*It).MaxElemComputeCapability;
			}
			cout << "Введите имя файла" << endl;
			string filename;
			cin >> filename;
			size_t filesize;
			ifstream file (filename, ios::in | ios::binary);
			if (file.is_open())
			{
				for(int i = 0; i < DIMENSIONS; ++i)
				{
					file.read(reinterpret_cast<char*>(&(GlobalIntStart[i])), sizeof(float));
					file.read(reinterpret_cast<char*>(&(GlobalIntEnd[i])), sizeof(float));
					//file.read((char*)GlobalIntStart[i],sizeof(float));
				}
				cout << "Файл содержит следующую область:" << endl;
				cout << "Начало области: " << endl;
				for(int i = 0; i < DIMENSIONS; ++i)
				{
					cout << GlobalIntStart[i] << " ; ";
				}
				cout << endl;
				cout << "Конец области: " << endl;
				for(int i = 0; i < DIMENSIONS; ++i)
				{
					cout << GlobalIntEnd[i] << " ; ";
				}
				cout << endl;
				file.seekg(0, ios::end);
				filesize = file.tellg();
				filesize -= DIMENSIONS * 2 * sizeof(float);
				file.seekg(DIMENSIONS * 2 * sizeof(float), ios::beg);
				size_t ElementsAmount = filesize / sizeof(float) / DIMENSIONS;
				bool inputFlag = true;
				while (inputFlag)
				{
					cout << "Вы хотите искать на определенном промежутке?" << endl;
					char rep;
					cin >> rep;
					if (rep == 'Y' || rep == 'y')
					{
						cout << "Введите интервалы для поиска" << endl;
						cout << "Вы хотите использовать одни и те же интервалы для поиска?" << endl;				
						cin >> rep;
						if (rep == 'Y' || rep == 'y')
						{
							cout << "Введите начало каждого из интервалов" << endl;
							cin >> ChoosedIntStart[0];

							cout << "Введите конец каждого из интервалов:" << endl;
							cin >> ChoosedIntEnd[0];
							bool check = false;
							for (int i = 0; i < DIMENSIONS; ++i)
							{
								if (ChoosedIntStart[i] < GlobalIntStart[i] || ChoosedIntEnd[i] > GlobalIntEnd[i])
								{
									check = true;
									break;
								} 
							}
							if(!check)
							{
								for (int i = 1; i < DIMENSIONS; ++i)
								{
									ChoosedIntStart[i] = ChoosedIntStart[0];
									ChoosedIntEnd[i] = ChoosedIntEnd[0];

								}								
							}
							else
							{
								cout << "Введенные промежутки неверны, повторите ввод" << endl;
								continue;
							}
							
						}
						else
						{
							for (int i = 0; i < DIMENSIONS; ++i)
							{
								bool check = false;
								do
								{
									if(check) cout << "Неверное значение, повторите ввод"<< endl;
									cout << "Начало " << i << " го интервала" << endl;
									cin >> ChoosedIntStart[i]; 
									if(ChoosedIntStart[i] <= GlobalIntStart[i] && ChoosedIntStart[i] >= GlobalIntEnd[i]) check = true;
								} while (check);
								
								check = false;
								do
								{
									if(check) cout << "Неверное значение, повторите ввод"<< endl;
									cout << "Конец " << i << " го интервала:" << endl;
									cin >> ChoosedIntEnd[i]; 
									if(ChoosedIntEnd[i] >= GlobalIntEnd[i] && ChoosedIntEnd[i] <= GlobalIntStart[i]) check = true;
								} while (check);
							}
						} 
						for(int i = 0; i < DIMENSIONS; ++i)
						{
							if(ChoosedIntEnd[i] <= ChoosedIntStart[i])
							{
								continue;
							}
						}
					}
					else
					{
						for(int i = 0; i < DIMENSIONS; ++i)
						{
							ChoosedIntStart[i] = GlobalIntStart[i];
							ChoosedIntEnd[i] = GlobalIntEnd[i];
						}
					}
					inputFlag = false;
				}
				cout << "Обработка.." << endl;
				while (ElementsAmount > 0)
				{
					findLocalExtrema(file, Devices, GpuWorkers, ElementsAmount, min);
					compareExtrema(GlobalExtrema, ChunckExtrema, min);
					//TODO: Need to show progress somehow in a way that won't hurt much on perfomance
				}
				file.close();
				if (GlobalExtrema.size() > 0)
				{
					cout << "Найдено " << GlobalExtrema.size() / DIMENSIONS << " экстремумов. Записываю в файл.. "  << endl;
					ofstream results("extrema.txt");
					if (results.is_open())
					{

								for(list<float>::iterator i = GlobalExtrema.begin(); i != GlobalExtrema.end(); ++i)
								{
									float x0 = *i;
									float x1 = *(++i);
									float x2 = *(++i);
									float x3 = *(++i);
									float x4 = *(++i);
									float x5 = *(++i);
									float x6 = *(++i);
									float x7 = *(++i);
									float x8 = *(++i);
									float x9 = *(++i);
									results << "x0 = " << x0 << ", x1 = " << x1 << ", x2 = " << x2 << ", x3 = " << x3 <<
										", x4 = " << x4 << ", x5 = " << x5 << ", x6 = " << x6 << ", x7 = " << x7
										<< ", x8 = " << x8 << ", x9 = " << x9 <<
										" где F(X) = " << Ft_10D(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9) << endl;
								}
						
					}
					cout << "Готово" << endl;
				}
				else
				{
					cout << "Экстремумов не найдено" << endl;
				}
			}
			else
			{
				cout << " Произошла ошибка при открытии файлы" << endl;
			}
		}
		else cout << " К сожалению ваш компьютер несовместим с данной программой" << endl;
	}
	else
	{
		cout << "Не найдено видеоадаптеров поддерживающих CUDA" << endl;
	}
	DeleteCriticalSection(&Locker);
	system("pause");
	return 0;
}
