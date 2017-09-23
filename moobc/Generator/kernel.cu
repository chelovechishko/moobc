#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <list>
using namespace std;
#define DIMENSIONS 10
#define ARC 13
CRITICAL_SECTION Locker;
ofstream fout;
struct Device  //краткая информацию о девайсах для рабочих потоков
{
	int SystemDeviceId; //идентификатор видеокарты для запуска
	int MaxElemComputeCapability; //число элементов которое видеокарта может обрабатывать одновременно
	int MaxThreadPerBlock; //максимальное число потоков на блок
	Device(int systemDeviceId,int maxElemComputeCapability, int maxThreadsPerBlock)
	{
		SystemDeviceId = systemDeviceId;
		MaxElemComputeCapability = maxElemComputeCapability;
		MaxThreadPerBlock = maxThreadsPerBlock;
	}
	Device(const Device& d)
	{
		SystemDeviceId = d.SystemDeviceId;
		MaxElemComputeCapability = d.MaxElemComputeCapability;
		MaxThreadPerBlock = d.MaxThreadPerBlock;
	}
};

struct Work  //вся информация которая необходима для запуска кернела находится тут
{
	int DeviceIdToPerform;    //идентификатор видеокарты ждя запуска
	dim3 BlockAm;
	dim3 ThreadPerBlock;
	size_t Elements_Am;       //число элементов в текущем задании
	float* x_arr;
	Work()
	{}
	Work(const int& deviceIdToPerform, const size_t& elems_am, const dim3& blockAm, const dim3& threadPerBlock,  float* x_array)
	{
		DeviceIdToPerform = deviceIdToPerform;
		BlockAm = blockAm;
		ThreadPerBlock = threadPerBlock;
		x_arr = x_array;
		Elements_Am = elems_am;
	}
};
size_t ElemsComputeCapabiity; //общее количество элементов (одного измерения 10и мерной точки) которое все видеоадаптеры могут обработать одновременно

__device__ __host__ float Polynomial(float x)
{
	return x + x;
}

__global__ void GeneratePoints(float* d_x_arr, int elem_amount)
{
	int gthreadid =  blockDim.x * blockIdx.x + threadIdx.x;
	if(gthreadid < elem_amount) //т.к массив может быть меньше чем число запущенных потоков, эта проверка нужна
	{
		d_x_arr[gthreadid] = Polynomial(d_x_arr[gthreadid]);
	}
}

void writeToFile(float* arr, int pointam)
{
	/*Все хост потоки, видеокарты которых закончили генерацию вызовут этот метод
	 так что создадим критическую секцию и запищем сгенерированный кусок в файл
	*/
	EnterCriticalSection(&Locker);             
	for (int i = 0; i < pointam; i++)
	{
		float t = arr[i];
		fout.write(reinterpret_cast<const char*>(&t), sizeof t); 
	}
	LeaveCriticalSection(&Locker);
}

void launchGeneration(Work& Task)
{
	float* d_point_arr;

	cudaSetDevice(Task.DeviceIdToPerform); 

	cudaMalloc((void**)&d_point_arr, Task.Elements_Am * sizeof(float));

	cudaMemcpy(d_point_arr, Task.x_arr,	Task.Elements_Am * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	GeneratePoints<<<Task.BlockAm, Task.ThreadPerBlock>>>(d_point_arr, Task.Elements_Am);

	cudaMemcpy(Task.x_arr, d_point_arr, Task.Elements_Am * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	writeToFile(Task.x_arr, Task.Elements_Am);
	cudaFree(d_point_arr);
}

void launch_GPU_computation(thread& Worker, Work Task)
{
	Worker = thread(launchGeneration, Task);
}

void wait_GPU_results(thread& Worker)
{
	if (Worker.joinable())  //если поток работает
	{
		Worker.join();   //ждем пока он закончит работу
	}
}

int main()
{
	setlocale(LOCALE_ALL, "rus");
	int DeviceCount;
	cudaGetDeviceCount(&DeviceCount);
	if(DeviceCount > 0)
	{
		list<Device> Devices;
		cudaDeviceProp props;
		for(int i = 0; i < DeviceCount; ++i)       //Наполняем список устройств
		{
			cudaGetDeviceProperties(&props, i);
			if ( props.major > ARC / 10 || ( props.major == ARC / 10 && props.minor >= ARC % 10 ) )
			{
				/*
				// т.к каждый поток рассчитывает ровно один элемент (1 измерение точки) то максимальное число элементов, которое
				   гпу может посчитать за раз равен значению ниже
				*/
				int ElemsCC =  props.multiProcessorCount * props.maxThreadsPerMultiProcessor;  
				Device WorkGpu(i,  ElemsCC , props.maxThreadsPerBlock);
				if(ElemsCC % DIMENSIONS != 0)
				{
					/*
						Тут критично чтобы это число было кратно 10ти, ибо
						во время генерации исходных точек кернел вызывается лишь когда точка полностью сформирована
						Если значение не кратно 10ти умненьшаем его до ближайшего
					*/
					WorkGpu.MaxElemComputeCapability -= ElemsCC % DIMENSIONS;
				}
				Devices.push_back(WorkGpu);
			}
		}
		if(Devices.size() > 0)
		{
		    InitializeCriticalSection(&Locker);
			thread* GpuWorkers = new thread[Devices.size()];    // инициализируем потоки, которые будут вызывать кернелы
			for(list<Device>::iterator It = Devices.begin(); It != Devices.end(); ++It)
			{
				ElemsComputeCapabiity += (*It).MaxElemComputeCapability;
			}
			cout << "Введите имя файла" << endl;
			string filename;
			cin >> filename;
			fout.open(filename, ios::binary);
			fout.clear();
			//границы интервалов и шаги------------------------------------------------------------------
			double intStart[DIMENSIONS];	//начала интервалов
			double intEnd[DIMENSIONS];		//конец интервалов
			double intStep[DIMENSIONS];		//шаги для прохода по интервалам
			bool inputFlag = true;
			char rep;				//флаг, показывающий одинаковые ли интервалы делать для разных измерений
			while (inputFlag)
			{
				cout << "Вы хотите использовать одни и те же интервалы для разных измерений? Y/N" << endl;
				
				cin >> rep;

				if (rep == 'Y' || rep == 'y')
				{
					cout << "Введите начало каждого из интервалов" << endl;
					cin >> intStart[0];

					cout << "Введите конец каждого из интервалов:" << endl;
					cin >> intEnd[0];
					if (intEnd[0] <= intStart[0])
					{
						cout << "Введенные промежутки неверны" << endl;
						continue;
					}

					cout << "Введите шаг каждого из интервалов:" << endl;
					cin >> intStep[0];
					if ((intEnd[0] - intStart[0] <= intStep[0]) || (intStep[0] < 0) || intStart[0] > intEnd[0])
					{
						cout << "Введенные шаги неверны" << endl;
						continue;
					}

					for (int i = 1; i < DIMENSIONS; ++i)
					{
						intStart[i] = intStart[0];
						intEnd[i] = intEnd[0];
						intStep[i] = intStep[0];
					}
				}
				else
				{
					for (int i = 0; i < DIMENSIONS; ++i)
					{
						cout << "Начало " << i << " го интервала" << endl;
						cin >> intStart[i];

						cout << "Конец " << i << " го интервала" << endl;
						cin >> intEnd[i];
						if (intEnd[i] <= intStart[i])
						{
							continue;
						}

						cout << "Шаг " << i << " го интервала" << endl;
						cin >> intStep[i];
						if ((intEnd[i] - intStart[i] <= intStep[i]) || (intStep[i] < 0) || intStart[i] > intEnd[i])
						{
							continue;
						}
					}
				}
				inputFlag = false;
			}
			/*
			Принцип работы:
			Цикл генерирует точки в введенных интервалах, и как только сгенерированных точек становится 
			достаточно чтобы заполнить выч. мощность видеокарты, создает поток который будет обрабатывать данный кусок.
			Пока на фоне уже начался рассчет первого куска, массив продолжает генерироваться, и как только вновь наберется необходимое количество точек,
			запускает рассчет на оставшихся видеокартах.
			Генерация массива происходит очень быстро, так что видеокарты не будет простаивать пока ждут своей очереди на вызов 

			*/
			for(int i = 0; i < DIMENSIONS; ++i)         //записываем в начало файла данные о крайних точках сгенерированной области
			{
				float t = Polynomial(intStart[i]);
				fout.write(reinterpret_cast<const char*>(&t), sizeof (float));
				t = Polynomial(intEnd[i]);
				fout.write(reinterpret_cast<const char*>(&t), sizeof (float)); 
			}
			cout << "Обработка..." << endl;
			int PointAm = 0;      //используется для подсчета текущего общего кол-ва точек
			int CurpointAm = 0;
			int thrdid = 0;       //идентификатор потока, который следует запустить
			double cur_pos[DIMENSIONS];   //текущии позиции генерируемых интервалов
			cur_pos[0] = intStart[0];
			float* arr = new float[ElemsComputeCapabiity]; //выделям памяти ровно под то кол-во элементов которое видеокарты обработают за раз одновременно
			float* offset = arr;
			list<Device>::iterator It = Devices.begin();
			while(cur_pos[0] <= intEnd[0])
			{
				cur_pos[1] = intStart[1];
				while (cur_pos[1] <= intEnd[1])
				{
					cur_pos[2] = intStart[2];
					while (cur_pos[2] <= intEnd[2])
					{
						cur_pos[3] = intStart[3];
						while (cur_pos[3] <= intEnd[3])
						{
							cur_pos[4] = intStart[4];
							while (cur_pos[4] <= intEnd[4])
							{
								cur_pos[5] = intStart[5];
								while (cur_pos[5] <= intEnd[5])
								{
									cur_pos[6] = intStart[6];
									while (cur_pos[6] <= intEnd[6])
									{
										cur_pos[7] = intStart[7];
										while (cur_pos[7] <= intEnd[7])
										{
											cur_pos[8] = intStart[8];
											while (cur_pos[8] <= intEnd[8])
											{
												cur_pos[9] = intStart[9];
												while (cur_pos[9] <= intEnd[9])
												{
													for(int i = 0; i < DIMENSIONS; ++i)
													{
														arr[PointAm++] = cur_pos[i];
													}
													CurpointAm += DIMENSIONS;
													if(CurpointAm == (*It).MaxElemComputeCapability) // Если число точек достаточно чтобы запустить вычисление
													{
														CurpointAm = 0;
														//Создаем задание для видеокарты
														Work Task((*It).SystemDeviceId, (*It).MaxElemComputeCapability, ceil((*It).MaxElemComputeCapability /(float)(*It).MaxThreadPerBlock), (*It).MaxThreadPerBlock, offset);
														//Проверяем завершила ли видеркарта свою предыдущую задачу
														if (GpuWorkers[thrdid].joinable())
														{
														//если нет ждем
															wait_GPU_results(GpuWorkers[thrdid]);
														}
														//запускаем поток, который вызовет кернел для рассчета
														launch_GPU_computation(GpuWorkers[thrdid], Task); 
														if(thrdid < Devices.size() - 1)
														{
															//если мы раздали задачи не всем видеокартам, увеличиваем идентификатор потока обработчика и
															//продвигаем следующую видеокарту вперед
															offset += (*It).MaxElemComputeCapability;
															thrdid++;
															It++; 
															
														}
														else
														{
															//иначе ставим все по нулям, включая индекс генерации, ибо мы генерим ровно столько элементов сколько могут все видеокарты за раз рассчитать
															
															for(int i = 0; i < Devices.size(); ++i) //ждем пока все гпу закончат эту итерацию
															{
																if (GpuWorkers[thrdid].joinable()) //ибо запись преобразованного куска идет сюда же
																{
																	wait_GPU_results(GpuWorkers[thrdid]);
																}
															}
															thrdid = 0;
															PointAm = 0;
															offset = arr;
															It = Devices.begin();
														}
													}
													cur_pos[9] += intStep[9];
												}
												cur_pos[8] += intStep[8];
											}
											cur_pos[7] += intStep[7];
										}
										cur_pos[6] += intStep[6];	
									}
									cur_pos[5] += intStep[5];
								}
								cur_pos[4] += intStep[4];
							}
							cur_pos[3] += intStep[3];
						}
						cur_pos[2] += intStep[2];
					}
					cur_pos[1] += intStep[1];
				}
				cur_pos[0] += intStep[0];
			}
			if(CurpointAm > 0) //Если у нас не набралось элементов на максимальную загрузку видеоадаптера
			{
				/*
				создаем задачу, для обработка данного куска
				*/
				Work Task((*It).SystemDeviceId, CurpointAm, ceil(CurpointAm / (float)(*It).MaxThreadPerBlock), (*It).MaxThreadPerBlock, offset);
				if (GpuWorkers[thrdid].joinable())
				{
					wait_GPU_results(GpuWorkers[thrdid]);
				}
				launch_GPU_computation(GpuWorkers[thrdid], Task); 
			}
			for(int i = 0; i < Devices.size(); ++i)
			{
				wait_GPU_results(GpuWorkers[i]);
			}
			delete arr;
			cout << "Готово" << endl;
		}
	}
	else
	{
		cout << "Не найдено видеоадаптеров поддерживающих CUDA" << endl;
	}
	DeleteCriticalSection(&Locker);
	fout.close();
	system("pause");
	return 0;
}

