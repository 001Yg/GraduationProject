#pragma warning(disable:4996)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include<direct.h>

#define N 1000  // #nodes
#define M (N*(N-1)/2)  // #Universal links
#define epsilon 0.00001 // a small value close to zero
#define PMAX 0.5 // p_ij <- U[0,PMAX]
#define PMAX_v 2 // inverse of PMAX
#define NET 10 // #considered networks
#define RUN 100 // #runs with given parameters
#define Resolution 0.01 // Step of noise amplitude
#define pro 0.1 // proportion of probe links
#define pro_v 10 // inverse of pro
#define SAMPLE 100000 // #samples in estimating AUC value
#define XNUMMAX 101 // Possibly maximum number of eta
#define p_value 0.01 // p-value
#define Huge 1000 // a huge number to destinguish score and ID
#define Huge2 1000000 // a huge number to destinguish score and ID
#define TIMES 2 // How many different noise parameters

long int DMAX;
int XNUM; // #different eta X��Ĵ�С
double detail[NET][RUN][XNUMMAX]; // details of results for AUC
double detail2[NET][RUN][XNUMMAX]; // details of results for AUPR
double detail3[NET][RUN][XNUMMAX]; // details of results for Precision
double detail4[NET][RUN][XNUMMAX]; // details of results for Precision@k
double detail5[NET][RUN][XNUMMAX]; // details of results for Recall@k
double detail6[NET][RUN][XNUMMAX]; // details of results for F1_measure@k
double detail7[NET][RUN][XNUMMAX]; // details of results for auc_precision
double detail8[NET][RUN][XNUMMAX]; // details of results for auc_mroc
double detail9[NET][RUN][XNUMMAX]; // details of results for ndcg
double detail10[NET][RUN][XNUMMAX]; // details of results for mcc
double p[N][N]; // linking likelihood matrix��N��N-1)/2���ߵ�likelihood
double s[N][N]; // similarity matrix 
double eta; // noise amplitude
double mean[XNUMMAX] = { 0.0 }; // average AUC value @eta
double mean2[XNUMMAX] = { 0.0 }; // average AUPR value @eta
double mean3[XNUMMAX] = { 0.0 }; // average Precision value @eta
double mean4[XNUMMAX] = { 0.0 }; // average Precision@k value @eta
double mean5[XNUMMAX] = { 0.0 }; // average Recall@k value @eta
double mean6[XNUMMAX] = { 0.0 }; // average F1_measure@k value @eta
double mean7[XNUMMAX] = { 0.0 }; // average auc_precision value @eta
double mean8[XNUMMAX] = { 0.0 }; // average auc_mroc value @eta
double mean9[XNUMMAX] = { 0.0 }; // average ndcg value @eta
double mean10[XNUMMAX] = { 0.0 }; // average mcc value @eta
double se[XNUMMAX] = { 0.0 }; // standard deviations of AUC values @eta 
double se2[XNUMMAX] = { 0.0 }; //  standard deviations of AUPR values @eta 
double se3[XNUMMAX] = { 0.0 }; // standard deviations of Precision values @eta
double se4[XNUMMAX] = { 0.0 }; // standard deviations of Precision@k values @eta 
double se5[XNUMMAX] = { 0.0 }; //  standard deviations of Recall@k values @eta 
double se6[XNUMMAX] = { 0.0 }; // standard deviations of F1_measure@k values @eta
double se7[XNUMMAX] = { 0.0 }; // standard deviations of auc_precision values @eta
double se8[XNUMMAX] = { 0.0 }; // standard deviations of auc_mroc values @eta 
double se9[XNUMMAX] = { 0.0 }; //  standard deviations of ndcg values @eta 
double se10[XNUMMAX] = { 0.0 }; // standard deviations of mcc values @eta
int m, mt, mp; // #links in the original network, training set and probe set  m=mt+mp�˹������У����Ǹ����㷨�ź������Ϊǰmp������missing link����������֪missing link������Щ������ͨ����֪����������mp����precision
int g[N * N / PMAX_v / 2]; // all links,probing links+training links
int gp[N * N / PMAX_v / pro_v]; // all probing links
int label[N * N]; // link labels: 1-training; -1-probing; 0-none 
int threshold; // threshold of losses
long rs[N * (N - 1) / 2]; // temporal array for ranking
int r[N * N / PMAX_v / pro_v]; // rankings of all probing links

double tpr_m[N * (N - 1) / 2] = { 0.0 };
double fpr_m[N * (N - 1) / 2] = { 0.0 };
double tpr_m_rand[N * (N - 1) / 2] = { 0.0 };
double tp_rand[N * (N - 1) / 2] = { 0.0 };
double tpr_m_norm[N * (N - 1) / 2] = { 0.0 };
int tp[N * (N - 1) / 2];
int fp[N * (N - 1) / 2];


int comp(const void* a, const void* b)
{
	return *(long*)b - *(long*)a;
}

double Random01() // uniform distribution [0,1]
{
	int r1, r2;
	double r;
	r1 = rand();
	r2 = rand();
	//rand������Χ��[0,RAND_MAX]
	//���˵�� 10rand���������Ӧ�ķ�ΧӦ��Ϊ[0.0, 10RAND_MAX]
	//	һ����˵��rand() % (b - a + 1) + a; �ͱ�ʾ a~b ֮���һ�����������
	//	����
	//	Ҫȡ��[a, b)�����������ʹ��(rand() % (b - a)) + a �����ֵ��a����b����
	//	Ҫȡ��[a, b]�����������ʹ��(rand() % (b - a + 1)) + a �����ֵ��a��b��,����rand()/RAND_MAX
	//	Ҫȡ��(a, b]�����������ʹ��(rand() % (b - a)) + a + 1 �����ֵ����a��b����
	r = (double)(r1 * (RAND_MAX + 1) + r2) / (double)DMAX;//Ϊʲô��������α�����Ҫ�������㣬��ֱ��rand����/RAND_MAX����������
	return r;
}

int RandomNum(int num) // A random (not so big) number in [0,num-1]
{
	int r1, r2;
	r1 = rand();
	r2 = rand();
	return (r1 * (RAND_MAX + 1) + r2) % num;
}

void GeneP() // generating the linking likelihood matrix
{
	int i, j;
	for (i = 0; i < N - 1; i++)
	{
		for (j = i + 1; j < N; j++)
		{
			p[i][j] = Random01() * PMAX;
		}
	}
	return;
}

void GeneG() // generating networks as well as training and probing sets
{
	int i, j, k;
	m = 0;
	for (i = 0; i < N - 1; i++)
	{
		for (j = i + 1; j < N; j++)
		{
			//������link��linking likelihood���������0~1���������link����network��
			if (Random01() <= p[i][j])
			{
				g[m] = i * N + j;
				m++;
			}
		}
	}
	//��ʼ��lebel���飬ȫ��Ϊ0
	memset(label, 0, sizeof(label));
	mp = 0;
	for (k = 0; k < m; k++)
	{
		//������Լ���ռ�ȣ�0.1�����������0~1������ô������link������Լ�������������Ǿ��ȷֲ����������ܿ��Ʋ��Ի��ı���Ϊ0.9��
		if (Random01() < pro)
		{
			//��ֵ�������Լ������飬��������link��network�ϵ�label��Ϊ-1����ʾ��Ϊ���Լ�
			gp[mp] = g[k];
			label[g[k]] = -1;
			mp++;
		}
		else
		{
			label[g[k]] = 1;
		}
	}
	mt = m - mp;
	return;
}


void GeneR()
{
	int i, j, k;
	int temp_id;
	k = 0;
	for (i = 0; i < N - 1; i++)
	{
		for (j = i + 1; j < N; j++)
		{
			//�õ�ÿһ��
			if (label[i * N + j] != 1)
			{
				//i*N+j��������(N-2)*N+(N-1)
				//����%Huge2���ܵõ�(i * N + j)���������Ա���linkԭ�����ڵ�λ����Ϣ(i * N + j)������Ҫ�Ƚϴ�С�����ֱ�Ӽ���(i * N + j)����
				//s[i][j]+��i*N+j��������s����ֵ��0-1֮��Ƚ�С�����ֱ�Ӽ���(i * N + j)����ʹ��(i * N + j)ռ�Ƚϴ�С��������λ�����
				//ʹ��Huge��Huge2���ų�(i * N + j)��������ʹ��s[i][j]��Ϊ��С�Ƚϵ���������
				rs[k] = floor(s[i][j] * Huge) * Huge2 + (i * N + j);
				k++;
			}
		}
	}
	qsort(rs, N * (N - 1) / 2, sizeof(long), comp);
	j = 0;
	for (k = 0; k < mp; k++)
	{
		//��ʱrs�����Ѿ��Ӵ�С����%Huge2��õ��ľ���(i * N + j)��Ҳ��������link��ԭ��λ��
		temp_id = rs[j] % Huge2;
		while (label[temp_id] != -1)
		{
			//���label����-1�����ⲻ�ǲ��Լ��е�Ԫ�أ���ô��������һ��
			j++;
			temp_id = rs[j] % Huge2;
		}
		//����ǲ��Լ��е�Ԫ�أ���ô�������λ�ø���r���飬r������rankings of all probing links
		r[k] = j;
		j++;
	}
	return;
}

double GetAUC()
{
	int i;
	double r_ave, au;
	r_ave = 0.0;
	for (i = 0; i < mp; i++)
	{
		r_ave = r_ave + r[i];
	}
	r_ave = r_ave / mp + 1;//�����Ǵ�0��ʼ�ģ�������ʵ������Ӧ�ô�1��ʼ����������Ҫ��1
	au = 1 - r_ave / (M - m) + (mp + 1) / 2 / (M - m);
	return au;
}

double GetAUPR()
{
	int i;
	double sum1, sum2, au;
	sum1 = 0.0;
	sum2 = 0.0;
	au = 0.0;
	r[mp] = M - mt;//r[0]~r[mp-1]�зŵ���universal set�Ĵ�С��|U-ET|��
	for (i = 0; i < mp; i++)
	{
		//���r[i]>=r[mp]Ҳ����˵���Լ��е�i���ߵ�rank��universal set�Ĵ�С���󣨰�����˵���Լ��бߵ�����Ӧ��С�ڵ���|U-ET|��
		if (r[i] >= r[mp])
		{
			r[i] = r[mp] - 1;
		}
		sum1 = sum1 + (i + 1.0) / (r[i] + 1.0);//�����Ǵ�0��ʼ�ģ�������ʵ������Ӧ�ô�1��ʼ����������Ҫ��1
		if (i < mp - 1)
		{
			sum2 = sum2 + (i + 1.0) / (r[i + 1]);
		}
	}
	au = (sum1 + sum2) / 2 / mp;
	return au;
}

//PrecisionҲ����balanced precision
double GetPrecision()
{
	int i;
	double bp;
	i = 0;
	//BP�ļ��㷽ʽ����i��missing links����С��|EP|(Ҳ����mp������BP=i/mp
	//The balanced precision (or simply precision) is computed as the proportion of TP among the top-P������mp�� ranked samples
	//�����i����Ԥ�⼯�е�tp����
	while (r[i] < mp)
	{
		i++;
	}
	bp = (double)i / (double)mp;
	return bp;
}

double GetPrecision_k(int k)
{
	int i;
	double bp_k;
	i = 0;
	//BP@k�ļ��㷽ʽ����i��missing links����С��k����precision@k=i/k
	//The balanced precision@k is computed as the proportion of TP among the top-k ranked samples
	//�����i����ǰk���е�tp����
	while (r[i] < k)
	{
		i++;
	}
	bp_k = (double)i / (double)k;
	return bp_k * 10;
}

double GetRecall_k(int k)
{
	int i;
	double recall_k;
	i = 0;
	//recall�ļ��㷽ʽ����i��missing links����С��k����BP@k=i/k
	// 
	//The recall is computed as the proportion of TP among the top-P������mp�� ranked samples
	while (r[i] < k)
	{
		i++;
	}
	recall_k = (double)i / (double)mp;
	return recall_k * 10;
}

double GetF1measure_k(int k)
{
	double measure;
	measure = (2 * GetPrecision_k(k) * GetRecall_k(k)) / (GetPrecision_k(k) + GetRecall_k(k));
	return measure;
}

double GetAUC_Precision()
{
	int k;
	double auc_precision = 0.0;
	for (k = 0; k < mp; k++)
	{
		auc_precision = (GetPrecision_k(k) + GetPrecision_k(k + 1)) / 1;
	}
	auc_precision = auc_precision / 2;
	return auc_precision;
}

double GetAUC_mROC()
{
	double auc_mroc = 0;
	memset(tpr_m, 0, sizeof(tpr_m));
	memset(fpr_m, 0, sizeof(fpr_m));
	memset(tpr_m_rand, 0, sizeof(tpr_m_rand));
	memset(tp_rand, 0, sizeof(tp_rand));
	memset(tp, 0, sizeof(tp));
	memset(fp, 0, sizeof(fp));
	memset(tpr_m_norm, 0, sizeof(tpr_m_norm));

	int i = 0;
	//����tp
	for (int k = 0; k < N * (N - 1) / 2 - mt; k++)//����k��1-S֮�䣬S��ָ��U-ET�𣿣�������������
	{
		while (r[i] < k)
		{
			i++;
		}
		tp[k] = i;
	}
	//����fp��tp��fp֮����k)
	for (int k = 0; k < N * (N - 1) / 2 - mt; k++)
	{
		fp[k] = k + 1 - tp[k];
	}
	for (int k = 0; k < N * (N - 1) / 2 - mt; k++)
	{
		tpr_m[k] = (double)log(1 + tp[k]) / (double)log(1 + mp);//mp��P
		fpr_m[k] = (double)log(1 + fp[k]) / (double)log(1 + N * (N - 1) / 2 - m);// N * (N - 1) / 2 - m��N
		tp_rand[k] = (double)fp[k] * mp / (double)(N * (N - 1) / 2 - m);
		tpr_m_rand[k] = (double)log(1 + tp_rand[k]) / (double)log(1 + mp);
		tpr_m_norm[k] = (double)(tpr_m[k] - tpr_m_rand[k]) / (double)(1 - tpr_m_rand[k]) * (1 - fpr_m[k]) + fpr_m[k];
	}
	int n = N * (N - 1) / 2 - mt;
	for (int k = 0; k < n - 2; k++)
	{
		auc_mroc += (double)(tpr_m_norm[k] + tpr_m_norm[k + 1]) * (fpr_m[k + 1] - fpr_m[k]) / 2; //���ε����֮�ͣ�Ҳ���������������
	}
	if (isinf(auc_mroc) or isnan(auc_mroc))
		auc_mroc = 0;
	return auc_mroc;
}

double GetNDCG()
{
	double dcg = 0, idcg = 0, ndcg = 0;
	for (int r = 1; r <= mp; r++)
	{
		idcg += 1 / (log(1 + r) / log(2));
	}
	for (int r_ = 0; r_ < mp; r_++)
	{
		dcg += 1 / (log(1 + r[r_]) / log(2));
	}
	if (idcg == 0)
		ndcg = 0;
	else
		ndcg = dcg / idcg;
	if (isinf(ndcg) or isnan(ndcg))
		ndcg = 0;
	return ndcg;
}

double GetMCC()
{
	int i;
	int tp, fp, tn, fn;
	double mcc;
	i = 0;
	//����tp
	while (r[i] < mp)
	{
		i++;
	}
	tp = i;
	fp = mp - i;
	fn = mp - i;
	tn = (M - m) - fp;
	mcc = (double)(tp * tn - fp * fn) / sqrt((long long)(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
	if (isinf(mcc) or isnan(mcc))
		mcc = 0;
	return mcc;
}

void GetDiscrimination(int yes, int e1, int e2, int loss, int temp_net, int temp_run, double detail_matric[][100][101], FILE* fp_matric)
{
	e2 = e1 + 1;
	yes = 0;
	while (yes == 0 && e2 < XNUM)
	{
		loss = 0;
		for (temp_net = 0; temp_net < NET; temp_net++)
		{
			for (temp_run = 0; temp_run < RUN; temp_run++)
			{
				if (detail_matric[temp_net][temp_run][e1] <= detail_matric[temp_net][temp_run][e2])
				{
					loss++;//�൱��p-value�����е�z 
				}
			}
		}
		//���z������ֵ��������eta2
		if (loss >= threshold)
		{
			e2++;
		}
		//����eta1��eta2 ��discrimination�ǴﵽԤ�ڵ�
		else
		{
			yes = 1;
		}
	}
	//�����е�eta����һ��������0��1��2....����ʵ�ʵ�����Ӧ����0.01��0.02...������������Ϊ�˷�����㣬������resolution�Ϳ�����
	fprintf(fp_matric, "%f %f\n", e1 * Resolution, e2 * Resolution);
}

void main()
{
	int temp_net;
	int temp_run;
	int temp_eta;
	int i, j;
	int yes;
	int e1, e2;//����ǿ��eta1��eta2������discriminating matrix
	int loss;
	double noise;
	int sign;
	double auc, aupr, precision, precision_k, recall_k, f1measure_k, auc_precision, auc_mroc, ndcg, mcc;
	FILE* fp1;
	FILE* fp2;
	FILE* fp3;
	FILE* fp4;
	FILE* fp5;
	FILE* fp6;
	FILE* fp7;
	FILE* fp8;
	FILE* fp9;
	FILE* fp10;
	FILE* fp11;
	FILE* fp12;
	FILE* fp13;
	FILE* fp14;
	FILE* fp15;
	FILE* fp16;
	FILE* fp17;
	FILE* fp18;
	FILE* fp19;
	FILE* fp20;
	FILE* fp21;
	FILE* fp22;
	FILE* fp23;
	FILE* fp24;
	FILE* fp25;
	FILE* fp26;
	FILE* fp27;
	FILE* fp28;
	FILE* fp29;
	FILE* fp30;

	//srand������rand���������ʹ�õģ�û��srand��time��NULL))��ôrand����ÿ��������ɵĶ���һ��
	srand(time(NULL)); 
	mkdir("G:\\jxs\\results");
	mkdir("G:\\jxs\\results\\auc");
	mkdir("G:\\jxs\\results\\aupr");
	mkdir("G:\\jxs\\results\\bp");
	mkdir("G:\\jxs\\results\\precision_k");
	mkdir("G:\\jxs\\results\\recall_k");
	mkdir("G:\\jxs\\results\\f1measure_k");
	mkdir("G:\\jxs\\results\\auc_precision");
	mkdir("G:\\jxs\\results\\auc_mroc");
	mkdir("G:\\jxs\\results\\ndcg");
	mkdir("G:\\jxs\\results\\mcc");

	fp1 = fopen("G:\\jxs\\results\\auc\\Z21auc_details-p01-1000.txt", "w");
	fp2 = fopen("G:\\jxs\\results\\auc\\Z21auc_statistics-p01-1000.txt", "w");
	fp3 = fopen("G:\\jxs\\results\\auc\\Z21auc_distinguish-p01-1000.txt", "w");
	fp4 = fopen("G:\\jxs\\results\\aupr\\Z21aupr_details-p01-1000.txt", "w");
	fp5 = fopen("G:\\jxs\\results\\aupr\\Z21aupr_statistics-p01-1000.txt", "w");
	fp6 = fopen("G:\\jxs\\results\\aupr\\Z21aupr_distinguish-p01-1000.txt", "w");
	fp7 = fopen("G:\\jxs\\results\\bp\\Z21bp_details-p01-1000.txt", "w");
	fp8 = fopen("G:\\jxs\\results\\bp\\Z21bp_statistics-p01-1000.txt", "w");
	fp9 = fopen("G:\\jxs\\results\\bp\\Z21bp_distinguish-p01-1000.txt", "w");
	fp10 = fopen("G:\\jxs\\results\\precision_k\\Z21precision_k_details-p01-1000-k1000.txt", "w");
	fp11 = fopen("G:\\jxs\\results\\precision_k\\Z21precision_k_statistics-p01-1000-k1000.txt", "w");
	fp12 = fopen("G:\\jxs\\results\\precision_k\\Z21precision_k_distinguish-p01-1000-k1000.txt", "w");
	fp13 = fopen("G:\\jxs\\results\\recall_k\\Z21recall_k_details-p01-1000-k1000.txt", "w");
	fp14 = fopen("G:\\jxs\\results\\recall_k\\Z21recall_k_statistics-p01-1000-k1000.txt", "w");
	fp15 = fopen("G:\\jxs\\results\\recall_k\\Z21recall_k_distinguish-p01-1000-k1000.txt", "w");
	fp16 = fopen("G:\\jxs\\results\\f1measure_k\\Z21f1measure_k_details-p01-1000-k1000.txt", "w");
	fp17 = fopen("G:\\jxs\\results\\f1measure_k\\Z21f1measure_k_statistics-p01-1000-k1000.txt", "w");
	fp18 = fopen("G:\\jxs\\results\\f1measure_k\\Z21f1measure_k_distinguish-p01-1000-k1000.txt", "w");
	fp19 = fopen("G:\\jxs\\results\\auc_precision\\Z21auc_precision_details-p01-1000.txt", "w");
	fp20 = fopen("G:\\jxs\\results\\auc_precision\\Z21auc_precision_statistics-p01-1000.txt", "w");
	fp21 = fopen("G:\\jxs\\results\\auc_precision\\Z21auc_precision_distinguish-p01-1000.txt", "w");
	fp22 = fopen("G:\\jxs\\results\\auc_mroc\\Z21auc_mroc_details-p01-1000.txt", "w");
	fp23 = fopen("G:\\jxs\\results\\auc_mroc\\Z21auc_mroc_statistics-p01-1000.txt", "w");
	fp24 = fopen("G:\\jxs\\results\\auc_mroc\\Z21auc_mroc_distinguish-p01-1000.txt", "w");
	fp25 = fopen("G:\\jxs\\results\\ndcg\\Z21ndcg_details-p01-1000.txt", "w");
	fp26 = fopen("G:\\jxs\\results\\ndcg\\Z21ndcg_statistics-p01-1000.txt", "w");
	fp27 = fopen("G:\\jxs\\results\\ndcg\\Z21ndcg_distinguish-p01-1000.txt", "w");
	fp28 = fopen("G:\\jxs\\results\\mcc\\Z21mcc_details-p01-1000.txt", "w");
	fp29 = fopen("G:\\jxs\\results\\mcc\\Z21mcc_statistics-p01-1000.txt", "w");
	fp30 = fopen("G:\\jxs\\results\\mcc\\Z21mcc_distinguish-p01-1000.txt", "w");

	DMAX = (RAND_MAX + 1) * (RAND_MAX + 1) - 1;
	XNUM = (int)1 + (TIMES * PMAX + epsilon) / Resolution;//�����ʽ����ô�õ��ģ�����������������
	threshold = NET * RUN * p_value;//threshold=10*100*0.01=10
	GeneP();//generating the link likehood matrix
	//���ѭ����Ŀ�����ظ�ִ�г������ڼ��㲻ͬ������Ͳ�ͬ�Ĳ����µ�AUC��AUPR�;���ֵ��
	//NET��һ��������������ʾ��Ҫִ�еĴ�������ÿһ��ѭ���У������ʹ�ò�ͬ������Ͳ�������������
	//���������������Ӧ�ı����С���������Ŀ����Ϊ��������ͬ��ģ�ͺͲ��������ܱ��֣��Ա�ѡ�����ŵ�ģ�ͺͲ�����
	for (temp_net = 0; temp_net < NET; temp_net++)
	{
		GeneG();// �������硢ѵ���������Լ�
		for (temp_run = 0; temp_run < RUN; temp_run++)
		{
			printf("NET=%d RUN=%d\n", temp_net, temp_run);
			eta = 0;
			temp_eta = 0;
			//etaС��1ʱ
			while (eta < TIMES * PMAX + epsilon)//eta<������������*������+0.00001����������������������
			{
				//ͨ��ĳ������Ϊeta���㷨��potential link���
				for (i = 0; i < N - 1; i++)//N=1000
				{
					for (j = i + 1; j < N; j++)
					{
						//rand()%2����0����1��sign����ֵ��-1����1
						sign = rand() % 2 * 2 - 1;//�����Ҫ����0~m-1��m�������е�һ��������������Ա��Ϊ��int num = rand() % m
						//nosie strength��-eta��eta�ľ��ȷֲ��õ���
						noise = Random01() * eta * sign;
						//scoring each potential link
						s[i][j] = p[i][j] + noise;
						if (s[i][j] < 0)
						{
							s[i][j] = 0;
						}
						if (s[i][j] > 1)
						{
							s[i][j] = 1;
						}
					}
				}
				//���ɲ��Լ���link������
				GeneR();
				//���metric value
				auc = GetAUC();
				aupr = GetAUPR();
				precision = GetPrecision();
				precision_k = GetPrecision_k(1000);
				recall_k = GetRecall_k(1000);
				f1measure_k = GetF1measure_k(1000);
				auc_precision = GetAUC_Precision();
				auc_mroc = GetAUC_mROC();
				ndcg = GetNDCG();
				mcc = GetMCC();

				fprintf(fp1, "%d %d %f %f\n", temp_net, temp_run, eta, auc);
				fprintf(fp4, "%d %d %f %f\n", temp_net, temp_run, eta, aupr);
				fprintf(fp7, "%d %d %f %f\n", temp_net, temp_run, eta, precision);
				fprintf(fp10, "%d %d %f %f\n", temp_net, temp_run, eta, precision_k);
				fprintf(fp13, "%d %d %f %f\n", temp_net, temp_run, eta, recall_k);
				fprintf(fp16, "%d %d %f %f\n", temp_net, temp_run, eta, f1measure_k);
				fprintf(fp19, "%d %d %f %f\n", temp_net, temp_run, eta, auc_precision);
				fprintf(fp22, "%d %d %f %f\n", temp_net, temp_run, eta, auc_mroc);
				fprintf(fp25, "%d %d %f %f\n", temp_net, temp_run, eta, ndcg);
				fprintf(fp28, "%d %d %f %f\n", temp_net, temp_run, eta, mcc);

				detail[temp_net][temp_run][temp_eta] = auc;
				detail2[temp_net][temp_run][temp_eta] = aupr;
				detail3[temp_net][temp_run][temp_eta] = precision;
				detail4[temp_net][temp_run][temp_eta] = precision_k;
				detail5[temp_net][temp_run][temp_eta] = recall_k;
				detail6[temp_net][temp_run][temp_eta] = f1measure_k;
				detail7[temp_net][temp_run][temp_eta] = auc_precision;
				detail8[temp_net][temp_run][temp_eta] = auc_mroc;
				detail9[temp_net][temp_run][temp_eta] = ndcg;
				detail10[temp_net][temp_run][temp_eta] = mcc;

				temp_eta++;
				//etaÿ��+0.01
				eta += Resolution;
			}
		}
	}
	//auc��eta)��aupr��eta)��precision��eta)
	fclose(fp1);
	fclose(fp4);
	fclose(fp7);
	fclose(fp10);
	fclose(fp13);
	fclose(fp16);
	fclose(fp19);
	fclose(fp22);
	fclose(fp25);
	fclose(fp28);

	for (temp_eta = 0; temp_eta < XNUM; temp_eta++)
	{
		mean[temp_eta] = 0;
		mean2[temp_eta] = 0;
		mean3[temp_eta] = 0;
		mean4[temp_eta] = 0;
		mean5[temp_eta] = 0;
		mean6[temp_eta] = 0;
		mean7[temp_eta] = 0;
		mean8[temp_eta] = 0;
		mean9[temp_eta] = 0;
		mean10[temp_eta] = 0;
		//һ����eta�£�����10��network��ÿ��metwork100�����е�ƽ��auc��aupr��bp��ֵ
		for (temp_net = 0; temp_net < NET; temp_net++)
		{
			for (temp_run = 0; temp_run < RUN; temp_run++)
			{
				mean[temp_eta] += detail[temp_net][temp_run][temp_eta];
				mean2[temp_eta] += detail2[temp_net][temp_run][temp_eta];
				mean3[temp_eta] += detail3[temp_net][temp_run][temp_eta];
				mean4[temp_eta] += detail4[temp_net][temp_run][temp_eta];
				mean5[temp_eta] += detail5[temp_net][temp_run][temp_eta];
				mean6[temp_eta] += detail6[temp_net][temp_run][temp_eta];
				mean7[temp_eta] += detail7[temp_net][temp_run][temp_eta];
				mean8[temp_eta] += detail8[temp_net][temp_run][temp_eta];
				mean9[temp_eta] += detail9[temp_net][temp_run][temp_eta];
				mean10[temp_eta] += detail10[temp_net][temp_run][temp_eta];
			}
		}
		mean[temp_eta] = mean[temp_eta] / NET / RUN;
		mean2[temp_eta] = mean2[temp_eta] / NET / RUN;
		mean3[temp_eta] = mean3[temp_eta] / NET / RUN;
		mean4[temp_eta] = mean4[temp_eta] / NET / RUN;
		mean5[temp_eta] = mean5[temp_eta] / NET / RUN;
		mean6[temp_eta] = mean6[temp_eta] / NET / RUN;
		mean7[temp_eta] = mean7[temp_eta] / NET / RUN;
		mean8[temp_eta] = mean8[temp_eta] / NET / RUN;
		mean9[temp_eta] = mean9[temp_eta] / NET / RUN;
		mean10[temp_eta] = mean10[temp_eta] / NET / RUN;
	}

	for (temp_eta = 0; temp_eta < XNUM; temp_eta++)
	{
		//��׼��
		se[temp_eta] = 0;
		se2[temp_eta] = 0;
		se3[temp_eta] = 0;
		se4[temp_eta] = 0;
		se5[temp_eta] = 0;
		se6[temp_eta] = 0;
		se7[temp_eta] = 0;
		se8[temp_eta] = 0;
		se9[temp_eta] = 0;
		se10[temp_eta] = 0;
		for (temp_net = 0; temp_net < NET; temp_net++)
		{
			for (temp_run = 0; temp_run < RUN; temp_run++)
			{
				se[temp_eta] += (detail[temp_net][temp_run][temp_eta] - mean[temp_eta]) * (detail[temp_net][temp_run][temp_eta] - mean[temp_eta]);
				se2[temp_eta] += (detail2[temp_net][temp_run][temp_eta] - mean2[temp_eta]) * (detail2[temp_net][temp_run][temp_eta] - mean2[temp_eta]);
				se3[temp_eta] += (detail3[temp_net][temp_run][temp_eta] - mean3[temp_eta]) * (detail3[temp_net][temp_run][temp_eta] - mean3[temp_eta]);
				se4[temp_eta] += (detail4[temp_net][temp_run][temp_eta] - mean4[temp_eta]) * (detail4[temp_net][temp_run][temp_eta] - mean4[temp_eta]);
				se5[temp_eta] += (detail5[temp_net][temp_run][temp_eta] - mean5[temp_eta]) * (detail5[temp_net][temp_run][temp_eta] - mean5[temp_eta]);
				se6[temp_eta] += (detail6[temp_net][temp_run][temp_eta] - mean6[temp_eta]) * (detail6[temp_net][temp_run][temp_eta] - mean6[temp_eta]);
				se7[temp_eta] += (detail7[temp_net][temp_run][temp_eta] - mean7[temp_eta]) * (detail7[temp_net][temp_run][temp_eta] - mean7[temp_eta]);
				se8[temp_eta] += (detail8[temp_net][temp_run][temp_eta] - mean8[temp_eta]) * (detail8[temp_net][temp_run][temp_eta] - mean8[temp_eta]);
				se9[temp_eta] += (detail9[temp_net][temp_run][temp_eta] - mean9[temp_eta]) * (detail9[temp_net][temp_run][temp_eta] - mean9[temp_eta]);
				se10[temp_eta] += (detail10[temp_net][temp_run][temp_eta] - mean10[temp_eta]) * (detail10[temp_net][temp_run][temp_eta] - mean10[temp_eta]);
			}
		}
		se[temp_eta] = sqrt(se[temp_eta] / NET / RUN);
		se2[temp_eta] = sqrt(se2[temp_eta] / NET / RUN);
		se3[temp_eta] = sqrt(se3[temp_eta] / NET / RUN);
		se4[temp_eta] = sqrt(se4[temp_eta] / NET / RUN);
		se5[temp_eta] = sqrt(se5[temp_eta] / NET / RUN);
		se6[temp_eta] = sqrt(se6[temp_eta] / NET / RUN);
		se7[temp_eta] = sqrt(se7[temp_eta] / NET / RUN);
		se8[temp_eta] = sqrt(se8[temp_eta] / NET / RUN);
		se9[temp_eta] = sqrt(se9[temp_eta] / NET / RUN);
		se10[temp_eta] = sqrt(se10[temp_eta] / NET / RUN);
	}
	for (temp_eta = 0; temp_eta < XNUM; temp_eta++)
	{
		fprintf(fp2, "%f %f %f\n", temp_eta * Resolution, mean[temp_eta], se[temp_eta]);
		fprintf(fp5, "%f %f %f\n", temp_eta * Resolution, mean2[temp_eta], se2[temp_eta]);
		fprintf(fp8, "%f %f %f\n", temp_eta * Resolution, mean3[temp_eta], se3[temp_eta]);
		fprintf(fp11, "%f %f %f\n", temp_eta * Resolution, mean4[temp_eta], se4[temp_eta]);
		fprintf(fp14, "%f %f %f\n", temp_eta * Resolution, mean5[temp_eta], se5[temp_eta]);
		fprintf(fp17, "%f %f %f\n", temp_eta * Resolution, mean6[temp_eta], se6[temp_eta]);
		fprintf(fp20, "%f %f %f\n", temp_eta * Resolution, mean7[temp_eta], se7[temp_eta]);
		fprintf(fp23, "%f %f %f\n", temp_eta * Resolution, mean8[temp_eta], se8[temp_eta]);
		fprintf(fp26, "%f %f %f\n", temp_eta * Resolution, mean9[temp_eta], se9[temp_eta]);
		fprintf(fp29, "%f %f %f\n", temp_eta * Resolution, mean10[temp_eta], se10[temp_eta]);
	}
	//ƽ��ֵ������
	fclose(fp2);
	fclose(fp5);
	fclose(fp8);
	fclose(fp11);
	fclose(fp14);
	fclose(fp17);
	fclose(fp20);
	fclose(fp23);
	fclose(fp26);
	fclose(fp29);
	//���discriminating matrix
	for (e1 = 0; e1 < XNUM - 1; e1++)
	{
		loss = 0;
		e2 = 0;
		yes = 0;
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail, fp3);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail2, fp6);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail3, fp9);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail4, fp12);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail5, fp15);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail6, fp18);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail7, fp21);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail8, fp24);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail9, fp27);
		GetDiscrimination(yes, e1, e2, loss, temp_net, temp_run, detail10, fp30);
	}
	//�ﵽ��ֵҪ���eta1��eta2
	fclose(fp3);
	fclose(fp6);
	fclose(fp9);
	fclose(fp12);
	fclose(fp15);
	fclose(fp18);
	fclose(fp21);
	fclose(fp24);
	fclose(fp27);
	fclose(fp30);
	return;
}
