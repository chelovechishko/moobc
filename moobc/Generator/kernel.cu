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
struct Device  //������� ���������� � �������� ��� ������� �������
{
	int SystemDeviceId; //������������� ���������� ��� �������
	int MaxElemComputeCapability; //����� ��������� ������� ���������� ����� ������������ ������������
	int MaxThreadPerBlock; //������������ ����� ������� �� ����
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

struct Work  //��� ���������� ������� ���������� ��� ������� ������� ��������� ���
{
	int DeviceIdToPerform;    //������������� ���������� ��� �������
	dim3 BlockAm;
	dim3 ThreadPerBlock;
	size_t Elements_Am;       //����� ��������� � ������� �������
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
size_t ElemsComputeCapabiity; //����� ���������� ��������� (������ ��������� 10� ������ �����) ������� ��� ������������� ����� ���������� ������������

__device__ __host__ float Polynomial(float x)
{
	return x + x;
}

__global__ void GeneratePoints(float* d_x_arr, int elem_amount)
{
	int gthreadid =  blockDim.x * blockIdx.x + threadIdx.x;
	if(gthreadid < elem_amount) //�.� ������ ����� ���� ������ ��� ����� ���������� �������, ��� �������� �����
	{
		d_x_arr[gthreadid] = Polynomial(d_x_arr[gthreadid]);
	}
}

void writeToFile(float* arr, int pointam)
{
	/*��� ���� ������, ���������� ������� ��������� ��������� ������� ���� �����
	 ��� ��� �������� ����������� ������ � ������� ��������������� ����� � ����
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
	if (Worker.joinable())  //���� ����� ��������
	{
		Worker.join();   //���� ���� �� �������� ������
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
		for(int i = 0; i < DeviceCount; ++i)       //��������� ������ ���������
		{
			cudaGetDeviceProperties(&props, i);
			if ( props.major > ARC / 10 || ( props.major == ARC / 10 && props.minor >= ARC % 10 ) )
			{
				/*
				// �.� ������ ����� ������������ ����� ���� ������� (1 ��������� �����) �� ������������ ����� ���������, �������
				   ��� ����� ��������� �� ��� ����� �������� ����
				*/
				int ElemsCC =  props.multiProcessorCount * props.maxThreadsPerMultiProcessor;  
				Device WorkGpu(i,  ElemsCC , props.maxThreadsPerBlock);
				if(ElemsCC % DIMENSIONS != 0)
				{
					/*
						��� �������� ����� ��� ����� ���� ������ 10��, ���
						�� ����� ��������� �������� ����� ������ ���������� ���� ����� ����� ��������� ������������
						���� �������� �� ������ 10�� ���������� ��� �� ����������
					*/
					WorkGpu.MaxElemComputeCapability -= ElemsCC % DIMENSIONS;
				}
				Devices.push_back(WorkGpu);
			}
		}
		if(Devices.size() > 0)
		{
		    InitializeCriticalSection(&Locker);
			thread* GpuWorkers = new thread[Devices.size()];    // �������������� ������, ������� ����� �������� �������
			for(list<Device>::iterator It = Devices.begin(); It != Devices.end(); ++It)
			{
				ElemsComputeCapabiity += (*It).MaxElemComputeCapability;
			}
			cout << "������� ��� �����" << endl;
			string filename;
			cin >> filename;
			fout.open(filename, ios::binary);
			fout.clear();
			//������� ���������� � ����------------------------------------------------------------------
			double intStart[DIMENSIONS];	//������ ����������
			double intEnd[DIMENSIONS];		//����� ����������
			double intStep[DIMENSIONS];		//���� ��� ������� �� ����������
			bool inputFlag = true;
			char rep;				//����, ������������ ���������� �� ��������� ������ ��� ������ ���������
			while (inputFlag)
			{
				cout << "�� ������ ������������ ���� � �� �� ��������� ��� ������ ���������? Y/N" << endl;
				
				cin >> rep;

				if (rep == 'Y' || rep == 'y')
				{
					cout << "������� ������ ������� �� ����������" << endl;
					cin >> intStart[0];

					cout << "������� ����� ������� �� ����������:" << endl;
					cin >> intEnd[0];
					if (intEnd[0] <= intStart[0])
					{
						cout << "��������� ���������� �������" << endl;
						continue;
					}

					cout << "������� ��� ������� �� ����������:" << endl;
					cin >> intStep[0];
					if ((intEnd[0] - intStart[0] <= intStep[0]) || (intStep[0] < 0) || intStart[0] > intEnd[0])
					{
						cout << "��������� ���� �������" << endl;
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
						cout << "������ " << i << " �� ���������" << endl;
						cin >> intStart[i];

						cout << "����� " << i << " �� ���������" << endl;
						cin >> intEnd[i];
						if (intEnd[i] <= intStart[i])
						{
							continue;
						}

						cout << "��� " << i << " �� ���������" << endl;
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
			������� ������:
			���� ���������� ����� � ��������� ����������, � ��� ������ ��������������� ����� ���������� 
			���������� ����� ��������� ���. �������� ����������, ������� ����� ������� ����� ������������ ������ �����.
			���� �� ���� ��� ������� ������� ������� �����, ������ ���������� ��������������, � ��� ������ ����� ��������� ����������� ���������� �����,
			��������� ������� �� ���������� �����������.
			��������� ������� ���������� ����� ������, ��� ��� ���������� �� ����� ����������� ���� ���� ����� ������� �� ����� 

			*/
			for(int i = 0; i < DIMENSIONS; ++i)         //���������� � ������ ����� ������ � ������� ������ ��������������� �������
			{
				float t = Polynomial(intStart[i]);
				fout.write(reinterpret_cast<const char*>(&t), sizeof (float));
				t = Polynomial(intEnd[i]);
				fout.write(reinterpret_cast<const char*>(&t), sizeof (float)); 
			}
			cout << "���������..." << endl;
			int PointAm = 0;      //������������ ��� �������� �������� ������ ���-�� �����
			int CurpointAm = 0;
			int thrdid = 0;       //������������� ������, ������� ������� ���������
			double cur_pos[DIMENSIONS];   //������� ������� ������������ ����������
			cur_pos[0] = intStart[0];
			float* arr = new float[ElemsComputeCapabiity]; //������� ������ ����� ��� �� ���-�� ��������� ������� ���������� ���������� �� ��� ������������
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
													if(CurpointAm == (*It).MaxElemComputeCapability) // ���� ����� ����� ���������� ����� ��������� ����������
													{
														CurpointAm = 0;
														//������� ������� ��� ����������
														Work Task((*It).SystemDeviceId, (*It).MaxElemComputeCapability, ceil((*It).MaxElemComputeCapability /(float)(*It).MaxThreadPerBlock), (*It).MaxThreadPerBlock, offset);
														//��������� ��������� �� ���������� ���� ���������� ������
														if (GpuWorkers[thrdid].joinable())
														{
														//���� ��� ����
															wait_GPU_results(GpuWorkers[thrdid]);
														}
														//��������� �����, ������� ������� ������ ��� ��������
														launch_GPU_computation(GpuWorkers[thrdid], Task); 
														if(thrdid < Devices.size() - 1)
														{
															//���� �� ������� ������ �� ���� �����������, ����������� ������������� ������ ����������� �
															//���������� ��������� ���������� ������
															offset += (*It).MaxElemComputeCapability;
															thrdid++;
															It++; 
															
														}
														else
														{
															//����� ������ ��� �� �����, ������� ������ ���������, ��� �� ������� ����� ������� ��������� ������� ����� ��� ���������� �� ��� ����������
															
															for(int i = 0; i < Devices.size(); ++i) //���� ���� ��� ��� �������� ��� ��������
															{
																if (GpuWorkers[thrdid].joinable()) //��� ������ ���������������� ����� ���� ���� ��
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
			if(CurpointAm > 0) //���� � ��� �� ��������� ��������� �� ������������ �������� �������������
			{
				/*
				������� ������, ��� ��������� ������� �����
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
			cout << "������" << endl;
		}
	}
	else
	{
		cout << "�� ������� �������������� �������������� CUDA" << endl;
	}
	DeleteCriticalSection(&Locker);
	fout.close();
	system("pause");
	return 0;
}

