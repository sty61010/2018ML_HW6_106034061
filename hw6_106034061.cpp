//
//  hw6_106034061.cpp
//  Program
//
//  Created by 曾靖渝 on 2018/11/27.
//  Copyright © 2018年 曾靖渝. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <set>
#include <time.h>
#include <random>
#define training_subset_number 9
#define element_number_in_subset 10
#define ETA 0.2
using namespace std;
using example=vector<double>;

double random_discrete_distribution(double px[element_number_in_subset])
{
    default_random_engine generator;
    discrete_distribution<double> distribution{px[0],px[1],px[2],px[3],px[4],px[5],px[6],px[7],px[8],px[9]};
    //Check the distribution
    const int n = 10000;
    const int nstars = 100;
    int p[10]={};
    for (int i=0; i < n; ++i) {
        double x = distribution(generator);
//        cout<<x<<endl;
        if ((x >= 0.0) && (x < 10.0)) ++p[int(x)];
    }
    for (int i=0; i<element_number_in_subset; ++i) {
        std::cout << i << ": ";
        std::cout << std::string(p[i]*nstars/n, '*') << std::endl;
    }
    return distribution(generator);
}
std::random_device rd;
std::default_random_engine gen=std::default_random_engine(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);
std::uniform_real_distribution<double> weight_range(-1.0,1.0);
class Perception_Learning//Perception Learning Classifier(Linear Seperabitlity)
{
private:
    vector<example> training_set;
    vector<string> class_labels;
    vector<double> weight;//Memorize all of Weights
    int c_x=0;//c(x)
    int h_x=0;//h(x)
    int attribute_number;
    int instance_number;
    int class_label_number;
    int modifying_number;//Memorize Modifying times
public:
    Perception_Learning()//Initial all of varibles
    {
        this->attribute_number=0;
        this->instance_number=0;
        this->class_label_number=0;
        modifying_number=0;
    }
    void initialization(int attribute_number, int instance_number, int class_label_number)//Initial all of varibles
    {
        this->attribute_number=attribute_number;
        this->instance_number=instance_number;
        this->class_label_number=class_label_number;
        modifying_number=0;
        for(int i=0;i<=attribute_number;i++)
        {
            double w=weight_range(gen);
            weight.push_back(0.2);
        }
    }

    void add_to_training_set(example ex, string class_label)
    {
        bool found=false;
        for(int i=0;i<class_labels.size();i++)
        {
            if(class_labels[i]==class_label)
            {
                found=true;
                ex.push_back(i);
                break;
            }
        }
        if(!found)
        {
            ex.push_back(class_labels.size());
            class_labels.push_back(class_label);
        }
        training_set.push_back(ex);
        learning(ex);
    }

    void print_training_set(void)
    {
        cout<<"->Training Set:"<<endl;
        for(auto ex:training_set)
        {
            print_example(ex);
        }
    }
    void print_example(example ex)
    {
        cout<<"->Exmaple:";
        for(int i=0;i<ex.size();i++)
        {
            if(i!=attribute_number)cout<<ex[i]<<" ";
            else cout<<class_labels[ex[i]];
            //            cout<<ex[i]<<" ";
        }
        cout<<endl;
    }
    void print_class_labels(void)
    {
        cout<<"->Class Labels:"<<endl;
        for(int i=0;i<class_labels.size();i++)
        {
            cout<<i<<": "<<class_labels[i]<<endl;
        }
    }
    void print_weight(void)//Print out all of Weight for Recording
    {
        cout<<"->Weights:";
        for(auto w:weight)
            cout<<w<<" ";
        cout<<endl;
    }
    int compute_ClassLabel(example ex)//Compute c(x)
    {
        cout<<"->c(x)="<<ex[attribute_number]<<endl;
        c_x=ex[attribute_number];
        return c_x;
    }
    int compute_H_ClassLabel(example ex)//Compute h(x)
    {
        double sum=0;
        
        for(int i=0;i<=attribute_number;i++)
        {
            if(i==0)
                sum+=weight[i];
            else
                sum+=ex[i-1]*weight[i];

        }
        if(sum>0)h_x=1;
        else h_x=0;
        cout<<"->h(x)="<<h_x<<endl;
        return h_x;
    }
    void learning(example ex)//Classifying
    {
        //        cout<<"Attribute Number:"<<attribute_number<<" ,Instance Number:"<<instance_number<<endl;
        cout<<"->Learning..........."<<endl;
        int hx=compute_H_ClassLabel(ex);
        int cx=compute_ClassLabel(ex);
        if(hx!=cx)modify_weight(ex);
        print_weight();
        cout<<"->Misclassified:"<<modifying_number<<" examples"<<endl;
    }
    void Classification(example ex)
    {
        int label=compute_H_ClassLabel(ex);
        cout<<class_labels[label];
    }
    void modify_weight(example ex)//Refresh Weights
    {
        cout<<"->Modify Weight"<<endl;
        for(int i=0;i<=attribute_number;i++)
        {
            if(i==0)
                weight[i]+=ETA*(double)1*(double)(c_x-h_x);
            else
                weight[i]+=ETA*(double)ex[i-1]*(double)(c_x-h_x);
        }
        modifying_number++;
    }
    void print_Modifying_Number(void)
    {
        cout<<"->Modify:"<<modifying_number<<" times"<<endl;
    }
};



class Induce_k_NN
{
private:
    vector<string> class_labels;
    vector<example> training_set;
    int attribute_number;
    int instance_number;
    int class_label_number;
public:
    Induce_k_NN()
    {
        this->attribute_number=0;
        this->instance_number=0;
        this->class_label_number=0;
    }
    void Initialization(int attribute_number, int instance_number, int class_label_number)
    {
        this->attribute_number=attribute_number;
        this->instance_number=instance_number;
        this->class_label_number=class_label_number;

    }
    void add_to_training_set(example ex, string class_label)
    {
        bool found=false;
        for(int i=0;i<class_labels.size();i++)
        {
            if(class_labels[i]==class_label)
            {
                found=true;
//                ex.push_back(i);
                break;
            }
        }
        if(!found)
        {
//            ex.push_back(class_labels.size());
            class_labels.push_back(class_label);
        }
        training_set.push_back(ex);
    }
    vector<example> get_training_set(void)
    {
        return training_set;
    }
    //1-NN Classifier=====================================================
    double calculate_error(example ex1, example ex2)//to calculate the distance between two examples
    {
//        cout<<"=====calculating==========================="<<endl;
        double sum=0;
        for(int i=0;i<ex1.size()-1;i++)
            sum+=pow((ex1[i]-ex2[i]),2);
        sum=sqrt(sum);
        return sum;
    }
    int classfier(example ex)//the classfier
    {
        int index=0;
        double error=1000000;
        for(auto s:training_set)
        {
            double new_error=calculate_error(s, ex);
            if(error>=new_error)
            {
                index=s[s.size()-1];
                error=new_error;
//                cout<<"->refresh:index:"<<s[s.size()-1]<<":"<<class_labels[s[s.size()-1]]<<":error="<<error<<" ,new_error="<<new_error<<endl;
            }
//            else cout<<"->No refresh:"<<s[s.size()-1]<<":"<<class_labels[s[s.size()-1]]<<":error"<<error<<" ,new_error="<<new_error<<endl;
            
        }
//        cout<<"->Choose index:"<<index<<endl;
        return index;
    }
    void print_training_set(void)
    {
        cout<<"->Training Set:"<<endl;
        for(auto ex:training_set)
        {
            print_example(ex);
        }
    }
    void print_example(example ex)
    {
        cout<<"->Exmaple:";
        for(int i=0;i<ex.size();i++)
        {
            if(i!=attribute_number)cout<<ex[i]<<" ";
            else cout<<class_labels[ex[i]];
//            cout<<ex[i]<<" ";
        }
        cout<<endl;
    }
    void print_class_labels(void)
    {
        cout<<"->Class Labels:"<<endl;
        for(int i=0;i<class_labels.size();i++)
        {
            cout<<i<<": "<<class_labels[i]<<endl;
        }
    }
};

class Adaboost
{
private:
    int attribute_number;
    int instance_number;
    int class_label_number;
    vector<string> class_labels;
    vector<example> training_set;
    Induce_k_NN Induce_Classifier_Textbook[training_subset_number], Induce_Classifier_Original[training_subset_number];
    Perception_Learning Linear_Classifier[training_subset_number];
    vector<double> px;
    vector<double> classifier_weight_textbook;
    vector<double> classifier_weight_original;
public:
    Adaboost(int attribute_number, int instance_number, int class_label_number)
    {
        this->attribute_number=attribute_number;
        this->instance_number=instance_number;
        this->class_label_number=class_label_number;
    }
    void add_to_training_set(example ex, string class_label)
    {
        bool found=false;
        for(int i=0;i<class_labels.size();i++)
        {
            if(class_labels[i]==class_label)
            {
                found=true;
                ex.push_back(i);
                break;
            }
        }
        if(!found)
        {
            ex.push_back(class_labels.size());
            class_labels.push_back(class_label);
        }
        training_set.push_back(ex);
    }
    //get example according to probability
    example get_example(void)
    {
        double d= dis(gen);
        for(int i=0;i<px.size();i++)
        {
            if(d<px[i])return training_set[i];
            d-=px[i];
        }
        return training_set[training_set.size()-1];
    }
    //Creating Subset
    vector<example> creating_subset(void)//check
    {
        vector<example>subset;
        for(int i=0;i<element_number_in_subset;i++)
        {
            example ex=get_example();
            subset.push_back(ex);
        }
        return subset;
    }
    //Subset Evaluation
    void subset_evaluation_textbook(int round)
    {
        double ERROR=0;
        double BETA=0;
        vector<bool>check;
        for(int i=0;i<training_set.size();i++)
        {
            example ex=training_set[i];
            int label=Induce_Classifier_Textbook[round].classfier(ex);
            if(label!=ex[attribute_number])
            {
                ERROR+=px[i];
                cout<<"->Example:"<<i<<":->Misclassify"<<endl;
            }
            check.push_back(label==ex[attribute_number]);
        }
        cout<<"->Error="<<ERROR<<endl;
        BETA=(double)ERROR/(double)(1-ERROR);
        cout<<"->BETA="<<BETA<<endl;
        if(BETA<-0.001||BETA>0.001)
        {
            for(int i=0;i<training_set.size();i++)
                if(check[i])px[i]*=BETA;
            double sum=0;
            for(int i=0;i<px.size();i++)sum+=px[i];
            for(int i=0;i<px.size();i++)px[i]/=sum;
        }
    }
    void subset_evaluation_original(int round)
    {
        double ERROR=0;
        double BETA=0;
        vector<bool>check;
        for(int i=0;i<training_set.size();i++)
        {
            example ex=training_set[i];
            int label=Induce_Classifier_Original[round].classfier(ex);
            if(label!=ex[attribute_number])
            {
                ERROR+=px[i];
                cout<<"->Example:"<<i<<":->Misclassify"<<endl;
            }
            check.push_back(label==ex[attribute_number]);
        }
        cout<<"->Error="<<ERROR<<endl;
        BETA=sqrt((double)ERROR/(double)(1-ERROR));
        cout<<"->BETA="<<BETA<<endl;
        if(BETA<-0.001||BETA>0.001)
        {
            for(int i=0;i<training_set.size();i++)
                if(check[i])px[i]*=BETA;
                else px[i]/=BETA;
            double sum=0;
            for(int i=0;i<px.size();i++)sum+=px[i];
            for(int i=0;i<px.size();i++)px[i]/=sum;
        }
    }
    //Subset Classification
    void subset_classification_textbook(int round)
    {
        cout<<"Induce Classifier:"<<round<<"->:Subset Classification(Textbook Version)->"<<endl;
        Induce_Classifier_Textbook[round].Initialization(attribute_number, instance_number, class_label_number);
        vector<example> subset=creating_subset();
        for(auto ex:subset)
            Induce_Classifier_Textbook[round].add_to_training_set(ex, class_labels[ex[ex.size()-1]]);
//        Induce_Classifier_Textbook[round].print_training_set();
        subset_evaluation_textbook(round);
    }
    void subset_classification_original(int round)
    {
        cout<<"Induce Classifier:"<<round<<"->:Subset Classification(Original Version)->"<<endl;
        Induce_Classifier_Original[round].Initialization(attribute_number, instance_number, class_label_number);
        vector<example> subset=creating_subset();
        for(auto ex:subset)
            Induce_Classifier_Original[round].add_to_training_set(ex, class_labels[ex[ex.size()-1]]);
//        Induce_Classifier_Original[round].print_training_set();
        subset_evaluation_original(round);
    }
    void subset_classification_linear(int round)
    {
        cout<<"Induce Classifier:"<<round<<"->:Linear Classification->"<<endl;
        Linear_Classifier[round].initialization(attribute_number, instance_number, class_label_number);
        
    }

    void Textbook_Version_Classification(void)
    {
        px=vector<double>(training_set.size(), (double)1/(double)training_set.size());
        //        print_all_probability();
        cout<<"Textbook Version....."<<endl;
        for(int i=1;i<training_subset_number;i++)
        {
            subset_classification_textbook(i);
            //            print_all_probability();
        }
        master_classifier_textbook();
    }
    void Original_Version_Classification(void)
    {
        px=vector<double>(training_set.size(), (double)1/(double)training_set.size());
        //        print_all_probability();
        cout<<"Original Version......"<<endl;
        for(int i=0;i<training_subset_number;i++)
        {
            subset_classification_original(i);
            //            print_all_probability();
        }
        master_classifier_original();
    }
    void Linear_Classification(void)
    {
        cout<<"Linear Classification....."<<endl;
        for(int i=0;i<training_subset_number;i++)
            subset_classification_linear(i);
    }
    //Master Classifier
    void master_classifier_textbook(void)
    {
        classifier_weight_textbook=vector<double>(training_subset_number,0.2);
        for(int i=0;i<training_set.size();i++)
        {
            cout<<">->>Start Master Classifier at:"<<i<<" Example....."<<endl;
            double positive=0, negative=0;
            example ex=training_set[i];
            for(int j=0;j<training_subset_number;j++)
            {
                int subset_classifier_label=Induce_Classifier_Textbook[j].classfier(ex);
                if(subset_classifier_label)positive+=classifier_weight_textbook[j];
                else negative+=classifier_weight_textbook[j];
            }
            int master_classifier_label=0;
            if(positive>negative)master_classifier_label=1;
            else master_classifier_label=0;
            cout<<"->Master Classifier Label:"<<master_classifier_label<<endl;
            cout<<"->Class Label:"<<ex[attribute_number]<<endl;
            if(ex[attribute_number]!=master_classifier_label)cout<<"->>>Misclassified......"<<endl;
            for(int j=0;j<training_subset_number;j++)
            {
                int subset_classifier_label=Induce_Classifier_Textbook[j].classfier(ex);
                classifier_weight_textbook[j]+=ETA*((double)ex[attribute_number]-(double)master_classifier_label)*(double)subset_classifier_label;
                cout<<"->For "<<j<<" subset Classifier....."<<endl;
            }
            print_subset_classifier_weight();

            cout<<">->>Finish Master Classifier at:"<<i<<" Example......"<<endl;
        }
    }
    void master_classifier_original(void)
    {
        classifier_weight_original=vector<double>(training_subset_number,0.2);
        for(int i=0;i<training_set.size();i++)
        {
            cout<<">->>Start Master Classifier at:"<<i<<" Example....."<<endl;
            double positive=0, negative=0;
            example ex=training_set[i];
            for(int j=0;j<training_subset_number;j++)
            {
                int subset_classifier_label=Induce_Classifier_Original[j].classfier(ex);
                if(subset_classifier_label)positive+=classifier_weight_original[j];
                else negative+=classifier_weight_original[j];
            }
            int master_classifier_label=0;
            if(positive>negative)master_classifier_label=1;
            else master_classifier_label=0;
            cout<<"->Master Classifier Label:"<<master_classifier_label<<endl;
            cout<<"->Class Label:"<<ex[attribute_number]<<endl;
            if(ex[attribute_number]!=master_classifier_label)cout<<"->>>Misclassified......"<<endl;
            for(int j=0;j<training_subset_number;j++)
            {
                int subset_classifier_label=Induce_Classifier_Original[j].classfier(ex);
                classifier_weight_original[j]+=ETA*((double)ex[attribute_number]-(double)master_classifier_label)*(double)subset_classifier_label;
                cout<<"->For "<<j<<" subset Classifier....."<<endl;
            }
            print_subset_classifier_weight();
            cout<<">->>Finish Master Classifier at:"<<i<<" Example......"<<endl;
        }
    }

    //Main Classification(including two version of adaboost algorithm)
    void Classification(void)
    {
//        Textbook_Version_Classification();
        cout<<"========================================================================================"<<endl;
        Original_Version_Classification();
    }
    void print_training_set(void)
    {
        cout<<"->Training Set:"<<endl;
        for(auto ex:training_set)
        {
            print_example(ex);
        }
    }
    void print_example(example ex)
    {
        cout<<"->Exmaple:";
        for(int i=0;i<ex.size();i++)
        {
            if(i!=attribute_number)cout<<ex[i]<<" ";
            else cout<<class_labels[ex[i]];
            //            cout<<ex[i]<<" ";
        }
        cout<<endl;
    }
    void print_class_labels(void)
    {
        cout<<"->Class Labels:"<<endl;
        for(int i=0;i<class_labels.size();i++)
        {
            cout<<i<<": "<<class_labels[i]<<endl;
        }
    }
    void print_all_probability(void)
    {
        cout<<"->All Probability:"<<endl;
        for(auto i:px)
            cout<<i<<" ";
        cout<<endl;
    }
    void print_subset_classifier_weight(void)
    {
        cout<<"->Subset Classifer Weight:"<<endl;
        for(int i=0;i<training_subset_number;i++)
            cout<<classifier_weight_original[i]<<" ";
        cout<<endl;
    }
};

int main(void)
{
//    double px[10]={0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.22,0.89};
//    cout<<random_discrete_distribution(px)<<endl;
    
    int attribute_number=4;
    int instance_number=90;
    int class_label_number=2;
    Adaboost Classifier(attribute_number, instance_number, class_label_number);
    Perception_Learning Linear_Classifier;
    Linear_Classifier.initialization(attribute_number, instance_number, class_label_number);
//    Linear_Classifier.print_weight();
    int i=0;
    while (instance_number--)
    {
        example ex;
        string class_label;
        double attribute_value;
        cout<<i++<<": ";
        for(int i=0;i<attribute_number;i++)
        {
            cin>>attribute_value;
            ex.push_back(attribute_value);
            getchar();
        }
        cin>>class_label;
        Classifier.add_to_training_set(ex, class_label);//to build a Training Set
//        Linear_Classifier.add_to_training_set(ex, class_label);
        
    }
//    Linear_Classifier.print_training_set();
//    Classifier.print_training_set();
//    Classifier.print_class_labels();
    Classifier.Classification();
    
    
    
    return 0;
}


/*
 
 5.1,3.5,1.4,0.2,Iris-setosa
 4.9,3.0,1.4,0.2,Iris-setosa
 4.7,3.2,1.3,0.2,Iris-setosa
 4.6,3.1,1.5,0.2,Iris-setosa
 4.6,3.4,1.4,0.3,Iris-setosa
 5.0,3.4,1.5,0.2,Iris-setosa
 4.4,2.9,1.4,0.2,Iris-setosa
 4.9,3.1,1.5,0.1,Iris-setosa
 5.4,3.7,1.5,0.2,Iris-setosa
 4.8,3.0,1.4,0.1,Iris-setosa
 4.3,3.0,1.1,0.1,Iris-setosa
 5.8,4.0,1.2,0.2,Iris-setosa
 5.7,4.4,1.5,0.4,Iris-setosa
 5.4,3.9,1.3,0.4,Iris-setosa
 5.7,3.8,1.7,0.3,Iris-setosa
 5.1,3.8,1.5,0.3,Iris-setosa
 5.4,3.4,1.7,0.2,Iris-setosa
 5.1,3.7,1.5,0.4,Iris-setosa
 4.6,3.6,1.0,0.2,Iris-setosa
 5.1,3.3,1.7,0.5,Iris-setosa
 5.6,2.5,1.9,1.1,Iris-setosa
 5.0,3.0,1.6,0.2,Iris-setosa
 5.0,3.4,1.6,0.4,Iris-setosa
 5.2,3.5,1.5,0.2,Iris-setosa
 5.2,3.4,1.4,0.2,Iris-setosa
 4.7,3.2,1.6,0.2,Iris-setosa
 4.8,3.1,1.6,0.2,Iris-setosa
 5.4,3.4,1.5,0.4,Iris-setosa
 5.5,4.2,1.4,0.2,Iris-setosa
 4.9,3.1,1.5,0.1,Iris-setosa
 
 
 
 5.0,3.2,1.2,0.2,Iris-setosa
 5.5,3.5,1.3,0.2,Iris-setosa
 4.9,3.1,1.5,0.1,Iris-setosa
 4.4,3.0,1.3,0.2,Iris-setosa
 5.1,3.4,1.5,0.2,Iris-setosa
 5.0,3.5,1.3,0.3,Iris-setosa
 4.5,2.3,1.3,0.3,Iris-setosa
 4.4,3.2,1.3,0.2,Iris-setosa
 5.1,3.8,1.9,0.4,Iris-setosa
 4.9,2.4,2.3,1.0,Iris-setosa
 5.1,3.8,1.6,0.2,Iris-setosa
 4.6,3.2,1.4,0.2,Iris-setosa
 5.3,3.7,1.5,0.2,Iris-setosa
 5.0,3.3,1.4,0.2,Iris-setosa
 5.0,2.3,3.3,1.0,Iris-setosa
 4.8,3.4,1.9,0.2,Iris-versicolor
 5.4,3.9,4.7,0.4,Iris-versicolor
 7.0,3.2,4.7,1.4,Iris-versicolor
 6.4,3.2,4.5,1.5,Iris-versicolor
 6.9,3.1,4.9,1.5,Iris-versicolor
 6.5,2.8,2.6,1.5,Iris-versicolor
 5.7,2.8,4.5,1.3,Iris-versicolor
 6.3,3.3,4.7,1.6,Iris-versicolor
 4.8,3.0,0.4,0.3,Iris-versicolor
 6.6,2.9,4.6,1.3,Iris-versicolor
 5.0,2.0,3.5,1.0,Iris-versicolor
 5.9,3.0,4.2,1.5,Iris-versicolor
 6.0,2.2,4.0,1.0,Iris-versicolor
 6.1,2.9,4.7,1.4,Iris-versicolor
 5.6,2.9,3.6,1.3,Iris-versicolor

 
 
 6.7,3.1,4.4,1.4,Iris-versicolor
 5.6,3.0,4.5,1.5,Iris-versicolor
 5.8,2.7,4.1,1.0,Iris-versicolor
 6.2,2.2,4.5,1.5,Iris-versicolor
 5.9,3.2,4.8,1.8,Iris-versicolor
 6.1,2.8,4.0,1.3,Iris-versicolor
 6.3,2.5,4.9,1.5,Iris-versicolor
 6.1,2.8,4.7,1.2,Iris-versicolor
 6.4,2.9,4.3,1.3,Iris-versicolor
 6.6,3.0,4.4,1.4,Iris-versicolor
 6.8,2.8,4.8,1.4,Iris-versicolor
 6.0,2.9,4.5,1.5,Iris-versicolor
 5.7,2.6,3.5,1.0,Iris-versicolor
 5.5,2.4,3.8,1.1,Iris-versicolor
 5.5,2.4,3.7,1.0,Iris-versicolor
 5.8,2.7,3.9,1.2,Iris-versicolor
 5.4,3.0,4.5,1.5,Iris-versicolor
 6.0,3.4,4.5,1.6,Iris-versicolor
 6.7,3.1,4.7,1.5,Iris-versicolor
 6.3,2.3,4.4,1.3,Iris-versicolor
 5.6,3.0,4.1,1.3,Iris-versicolor
 5.5,2.6,4.4,1.2,Iris-versicolor
 6.1,3.0,4.6,1.4,Iris-versicolor
 5.8,2.6,4.0,1.2,Iris-versicolor
 5.6,2.7,4.2,1.3,Iris-versicolor
 5.7,3.0,4.2,1.2,Iris-versicolor
 5.7,2.9,4.2,1.3,Iris-versicolor
 6.2,2.9,4.3,1.3,Iris-versicolor
 5.1,2.5,3.0,1.1,Iris-versicolor
 5.7,2.8,4.1,1.3,Iris-versicolor
 */

/*
 
 
 //    example random_get_example(void)
 //    {
 //        srand(time(NULL));
 //        double r=rand()%training_set.size();
 //        auto it=training_set.begin();
 //        for(;r>0;r--)it++;
 //        return *it;
 //    }
 //    example random_get_example_according_to_probability(double px[element_number_in_subset],set<example>last_subset)///check
 //    {
 //        double r=random_discrete_distribution(px);
 //        auto it=last_subset.begin();
 //        for(;r>0;r--)it++;
 //        return *it;
 //    }
 //    vector<example> creating_first_subset(void)
 //    {
 //        vector<example> first_subset;
 ////        Induce_k_NN subset_classifier(attribute_number, instance_number, class_label_number);
 //        for(int i=0;i<element_number_in_subset;i++)
 //        {
 //            example ex=random_get_example();
 //            first_subset.push_back(ex);
 ////            subset_classifier.add_to_training_set(ex, class_labels[ex[ex.size()-1]]);
 //        }
 ////        subset_classifier.print_training_set();
 //        return first_subset;
 //    }
 
 //    void first_subset_classifitcation(void)
 //    {
 //        cout<<"0:Subset Classification============"<<endl;
 //        Induce_Classifier[0].Initialization(attribute_number, instance_number, class_label_number);
 ////        Induce_Classifier[0].print_probability();
 //        vector<example> subset=creating_subset(Induce_Classifier[0].get_probability(), training_set);
 //        for(auto ex:subset)
 //            Induce_Classifier[0].add_to_training_set(ex, class_labels[ex[ex.size()-1]]);
 ////        Induce_Classifier[0].print_training_set();
 //    }
 
 //void memory_weight(int round, vector<double> &px)
 //{
 //    for(int i=0;i<element_number_in_subset;i++)
 //        this->px[round][i]=px[i];
 //}
 //void print_all_probability(void)
 //{
 //    for(int i=0;i<training_subset_number;i++)
 //    {
 //        for(int j=0;j<element_number_in_subset;j++)
 //            cout<<px[i][j]<<" ";
 //        cout<<endl;
 //    }
 //}
 
 */


