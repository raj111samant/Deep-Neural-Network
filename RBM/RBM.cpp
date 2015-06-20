#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

		unsigned short int train_X[8][4] = 
		{
   		{0, 0, 0, 0},
    		{0, 0, 1, 0},
    		{0, 1, 0, 0},
   		{0, 1, 1, 0},
   		{1, 0, 0, 0},
    		{1, 0, 1, 0},
		{1, 1, 0, 0},
    		{1, 1, 1, 0}
		};

  		unsigned short int test_X[2][4] = 
		{
    		{1, 0, 1, 0},
    		{0, 0, 1, 0}
  		};

float Uniform(float min, float max) {
  return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

unsigned short int Binomial(unsigned short int n, float p) 
{
	if(p < 0 || p > 1) 
		return 0;
  
	unsigned short int c = 0;
  	float r;
  
  	for(unsigned short int i=0; i<n; i++) 
	{
    		r = rand() / (RAND_MAX + 1.0);
    		if (r < p) 
			c++;
  	}
	if(c>floor(n/2))
		return 1;
	else
		return 0;
}

float Sigmoid(float x) 
{
  return 1.0 / (1.0 + exp(-x));
}


class RBM
{
	private:
		unsigned short int Number_Visible_Neurons;
		unsigned short int Number_Hidden_Neurons;

		float **Weights;
  		float *Hidden_Layer_Bias;
		float *Visible_Layer_Bias;
		
		float Learning_Rate;
		unsigned int Number_Training_Epochs;
		unsigned int Number_Training_Data;
		unsigned short int Contrastive_Divergence_k;
		
	
	public:
		~RBM();
		

		void InitializeNetworkParameters(unsigned short int nunber_visible, unsigned short int number_hidden);
		void InitializeTrainingParameters(unsigned int training_number, float learning_rate, unsigned int training_epochs, unsigned short int contrastive_divergence_k);
		void PrintTrainingParameterDetails();
		void PrintNetworkParameterDetails();
		void TrainNetwork(); 
		void ReconstructInput(unsigned short int *input, float *reconstructed_v);
		void TestNetwork();

		
		void Sample_H_V(unsigned short int *v_binary, float *h_continuous, unsigned short int *h_binary);
		void Sample_V_H(unsigned short int *h_binary, float *v_continuous, unsigned short int *v_binary); 
		float PropogateUp(unsigned short int *v_binary, float *weights, float h_bias); 
		float PropogateDown(unsigned short int *h_binary, unsigned short int i, float v_bias); 
		void Gibbs_H_V_H(unsigned short int *h_binary, float *v_continuous, unsigned short int *v_binary, float *h_continuous, unsigned short int *h_binary2); 
};

RBM::~RBM() 
{
	for(unsigned int i=0; i<Number_Hidden_Neurons; i++) 
		delete[] Weights[i];
	delete[] Weights;
	delete[] Hidden_Layer_Bias;
	delete[] Visible_Layer_Bias;
}

void RBM::InitializeNetworkParameters(unsigned short int number_visible, unsigned short int number_hidden)
{
	unsigned short int i = 0;
	Number_Visible_Neurons = number_visible;
	Number_Hidden_Neurons = number_hidden;

	Weights = new float*[Number_Hidden_Neurons];
	for(i=0; i<number_hidden; i++) 
		Weights[i] = new float[Number_Visible_Neurons];

float a = 1.0 / Number_Visible_Neurons;
	for(i=0; i<number_hidden; i++)
		for(int j=0; j<number_visible; j++)
			Weights[i][j] = Uniform(-a, a);//0.1*(2*rand() - 1);

	Hidden_Layer_Bias = new float[number_hidden];
	for(i=0; i<number_hidden; i++) 
		Hidden_Layer_Bias[i] = Uniform(-a, a);

    	Visible_Layer_Bias = new float[number_visible];
    	for(i=0; i<number_visible; i++) 
		Visible_Layer_Bias[i] = Uniform(-a, a);
}

void RBM::InitializeTrainingParameters(unsigned int training_number, float learning_rate, unsigned int training_epochs, unsigned short int contrastive_divergence_k)
{
	Number_Training_Data = training_number;
	Learning_Rate = learning_rate;
	Number_Training_Epochs = training_epochs;
	Contrastive_Divergence_k = contrastive_divergence_k;
}

void RBM::PrintNetworkParameterDetails()
{
	cout<<"| 1.1   | Info | Number of Visible Neurons - "<<Number_Visible_Neurons<<"\n";
	cout<<"| 1.2   | Info | Number of Hidden Neurons - "<<Number_Hidden_Neurons<<endl;
}

void RBM::PrintTrainingParameterDetails()
{
	cout<<"| 2.1   | Info | Number of Number Training Data - "<<Number_Training_Data<<"\n";
	cout<<"| 2.2   | Info | Learning Rate - "<<Learning_Rate<<"\n";
	cout<<"| 2.3   | Info | Number of Training Epochs - "<<Number_Training_Epochs<<endl;
}

void RBM::TrainNetwork()
{
	float *temp_input = new float [Number_Visible_Neurons];
	float *error = new float[Number_Training_Epochs];
	
	float *ph_continuous = new float[Number_Hidden_Neurons];
	unsigned short int *ph_binary = new unsigned short int[Number_Hidden_Neurons];
	float *v_continuous = new float[Number_Visible_Neurons];
  	unsigned short int *v_binary = new unsigned short int[Number_Visible_Neurons];
  	float *h_continuous = new float[Number_Hidden_Neurons];
  	unsigned short int *h_binary = new unsigned short int[Number_Hidden_Neurons];

	unsigned int epoch = 1;
	do
	{
		
		for(unsigned int i=0; i<Number_Training_Data; i++) 
		{
			if((epoch%1000) == 0)			
				cout<<"\e[A \r\r"<<"| 3.1   | Info | Running epoch number - "<<epoch<<endl;
			
			Sample_H_V(train_X[i], ph_continuous, ph_binary);
			for(unsigned short int step=0; step<Contrastive_Divergence_k; step++) 
			{
				if(step == 0) 
				{
					Gibbs_H_V_H(ph_binary, v_continuous, v_binary, h_continuous, h_binary);
				} 
				else 
				{
					Gibbs_H_V_H(h_binary, v_continuous, v_binary, h_continuous, h_binary);
				}
			}

			for(unsigned short int j=0; j<Number_Hidden_Neurons; j++) 
			{
				for(unsigned short int k=0; k<Number_Visible_Neurons; k++) 
				{
			 		//W[i][k] += lr * (ph_binary[i] * train_X[j] - h_binary[i] * v_binary[j]);
					Weights[j][k] += Learning_Rate * (ph_continuous[j] * train_X[i][k] - h_continuous[j] * v_continuous[k]);
				}
				//Hidden_Layer_Bias[j] += Learning_Rate * (ph_binary[j] - h_binary[j]);
				Hidden_Layer_Bias[j] += Learning_Rate * (ph_continuous[j] - h_continuous[j]);
			}

			for(unsigned short int j=0; j<Number_Visible_Neurons; j++) 
			{
				//Hidden_Layer_Bias[j] += Learning_Rate * (train_X[i][j] - v_binary[j]);
				Hidden_Layer_Bias[j] += Learning_Rate * (train_X[i][j] - v_continuous[j]);
			}
		}
	epoch++;
	}while(epoch <= Number_Training_Epochs);
}

void RBM::Sample_H_V(unsigned short int *v_binary, float *h_continuous, unsigned short int *h_binary) 
{
	for(unsigned short int i=0; i<Number_Hidden_Neurons; i++) 
	{
    		h_continuous[i] = PropogateUp(v_binary, Weights[i], Hidden_Layer_Bias[i]);
    		h_binary[i] = Binomial(5, h_continuous[i]);
  	}
}

void RBM::Sample_V_H(unsigned short int *h_binary, float *v_continuous, unsigned short int *v_binary) 
{
	for(unsigned short int i=0; i<Number_Visible_Neurons; i++) 
	{
    		v_continuous[i] = PropogateDown(h_binary, i, Visible_Layer_Bias[i]);
    		v_binary[i] = Binomial(5, v_continuous[i]);
  	}
}

float RBM::PropogateUp(unsigned short int *v_binary, float *weights, float h_bias) 
{
  	float temp = 0.0;
  	for(unsigned short int j=0; j<Number_Visible_Neurons; j++) 
	{
    		temp += weights[j] * v_binary[j];
  	}
  	temp += h_bias;
  	return Sigmoid(temp);
}

float RBM::PropogateDown(unsigned short int *h_binary, unsigned short int i, float v_bias) 
{
	float temp = 0.0;
  	for(unsigned short int j=0; j<Number_Hidden_Neurons; j++) 
	{
    		temp += Weights[j][i] * h_binary[j];
  	}
  	temp += v_bias;
 	return Sigmoid(temp);
}

void RBM::Gibbs_H_V_H(unsigned short int *h_binary, float *v_continuous, unsigned short int *v_binary, float *h_continuous, unsigned short int *h_binary2) 
{
  	Sample_V_H(h_binary, v_continuous, v_binary);
  	Sample_H_V(v_binary, h_continuous, h_binary2);
}

void RBM::ReconstructInput(unsigned short int *input, float *reconstructed_v) 
{
  	float *hidden_output = new float[Number_Hidden_Neurons];
  	float temp;

  	for(unsigned short int i=0; i<Number_Hidden_Neurons; i++) 
	{
    		hidden_output[i] = PropogateUp(input, Weights[i],	Hidden_Layer_Bias[i]);
  	}

  	for(unsigned short int i=0; i<Number_Visible_Neurons; i++) 
	{
    		temp = 0.0;
    		for(int j=0; j<Number_Hidden_Neurons; j++) 
		{
     		 temp += Weights[j][i] * hidden_output[j];
    		}
    		temp += Visible_Layer_Bias[i];

    		reconstructed_v[i] = Sigmoid(temp);
		reconstructed_v[i] = Binomial(5,reconstructed_v[i]);
		
		
  	}

  delete[] hidden_output;
}

void RBM::TestNetwork()
{
	cout<<"| 4.1   | Info | Testing results..."<<endl;
	float reconstructed_X[2][4];
	for(unsigned short int i=0; i<2; i++) 
	{
		cout<<"| 4.1."<<i+1<<" | Info | Input - ";
    		ReconstructInput(test_X[i], reconstructed_X[i]);
    		for(int j=0; j<Number_Visible_Neurons; j++) 
		{	
			cout<<test_X[i][j]<<" ";
      		
    		}
		cout<<"Output - ";
    		for(int j=0; j<Number_Visible_Neurons; j++) 
		{
			cout<<reconstructed_X[i][j]<<" ";
    		}
    		cout << endl;
  }
}

int main()
{
	srand(time(NULL));
	RBM Nexus;

	float learning_rate = 0.02;
	unsigned int training_epochs = 200000;
	unsigned short int contrastive_divergence_k = 10;

	unsigned int training_number = 6;
	unsigned short int test_number = 2;
	unsigned short int n_visible = 4;
	unsigned short int n_hidden = 10;

	Nexus.InitializeNetworkParameters(n_visible, n_hidden);
	cout<<"| 1     | Info | Network Initialized..."<<endl;
	
	Nexus.PrintNetworkParameterDetails();

	Nexus.InitializeTrainingParameters(training_number, learning_rate, training_epochs, contrastive_divergence_k);
	cout<<"| 2     | Info | Network Training Parameters Initialized..."<<endl;
	
	Nexus.PrintTrainingParameterDetails();
	
	cout<<"| 3     | Info | Starting Training..."<<endl;
	Nexus.TrainNetwork();
	
	cout<<"| 4     | Info | Starting Testing..."<<endl;
	Nexus.TestNetwork();
	cout<<"|       | Info | Program Ended..."<<endl;
	return 0;
}
