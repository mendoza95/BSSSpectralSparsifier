#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <string>
#include "Eigen/Eigenvalues" //Library eigen need to be downloaded and added in the same directory of this file

using namespace std;
using namespace Eigen;


int n, d, m;
double epsilon, delta;
double sigma = 1;
typedef vector<VectorXd> Puntos;
typedef pair< double, int > eigendata;
Puntos puntos;


//This method writes a matrix A in a csv file
void write_matrix(MatrixXd A, string filename){
    ofstream myfile;
    myfile.open (filename.c_str());
    for(int i = 0 ; i < A.rows() ; i ++){
        for(int j = 0 ; j < A.cols() ; j ++){
            myfile << A(i,j)<<",";
        }
        myfile << "\n";
    }
    myfile.close();
}

//This method returns gaussian similarity between two points
double gaussian_similarity(VectorXd x, VectorXd y){
    double s = exp(-pow((x-y).norm(),2)/pow(sigma,2));
    //cout<<s<<endl;
    return s;
}


MatrixXd build_reweightedW(MatrixXd W, VectorXd s){
    MatrixXd W1 = MatrixXd::Zero(m,m);
    for(int i = 0 ; i < m ; i++){
        W1(i,i) = W(i,i)*s(i);
    }
    return W1;
}

//This method construct the adjacency (similarity) matrix for a given set of points
MatrixXd build_Adjacency_matrix(){
    MatrixXd A = MatrixXd::Zero(n,n);
    double e;
    for(int i = 0 ; i < n; i++){
        for(int j = i+1 ; j < n ; j++){
            e = gaussian_similarity(puntos[i], puntos[j]);
            A(i,j) = e;
            A(j,i) = e;
        }
    }
    //cout<<"Adjacency matrix"<<endl;
    //cout<<A<<endl;
    return A;
}

//This method construct the degree diagonal Matrix
MatrixXd build_D(){
    MatrixXd D = MatrixXd::Zero(n,n);
    for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
            if(i != j){
                D(i,i) = D(i,i)+gaussian_similarity(puntos[i], puntos[j]);
            }
        }
    }
    //cout<<"Degree matrix"<<endl;
    //cout<<D<<endl;
    return D;
}

//This method construct the degree diagonal matrix D to the power of 1/sqrt(d(u))
MatrixXd build_Dsq(MatrixXd D){
    MatrixXd Dsq = MatrixXd::Zero(n,n);
    for(int i = 0 ; i < n ; i++){
        Dsq(i,i) = 1/sqrt(D(i,i));
    }
    return Dsq;
}

//This method construct the normalized Laplacian
MatrixXd build_NormalizedLaplacian(MatrixXd B, MatrixXd W, MatrixXd Dsq){
    MatrixXd L = B.transpose()*W*B;
    return Dsq*L*Dsq;
}

//This method construct the weighted edge degree matrix
MatrixXd build_W(){
    MatrixXd W = MatrixXd::Zero(m,m);
    int k = 0;
    for(int i = 0 ; i < n-1 ; i++){
        for(int j = i+1 ; j < n ; j++){
            W(k,k) = gaussian_similarity(puntos[i], puntos[j]);
            k ++;
        }
    }
    return W;
}

//This method construct the signed incedency matrix B
MatrixXd build_B(){
    MatrixXd B = MatrixXd::Zero(m,n);
    int k = 0;
    for(int i = 0 ; i < n-1 ; i++){
        for(int j = i+1 ; j < n ; j++){
            B(k,i) = 1;
            B(k,j) = -1;
            k ++;
        }
    }
    return B;
}


//This method construct the square root of the weighted edge matrix W
MatrixXd build_Wsq(MatrixXd W){
    MatrixXd Wsq = MatrixXd::Zero(m,m);
    for(int i = 0 ; i < m ; i++){
        Wsq(i,i) = sqrt(W(i,i));
    }
    //write_matrix(Wsq, "EdgeMatrixSQRT.csv");
    return Wsq;
}

//From now on we define the functions for the spectral sparsifier algorithm of Batson, Spielman and Srivastava


//This method retun a matrix V such that VV^T is the identity matrix over the image of L_G (see "Twice-Ramanujan Sparsifiers")
MatrixXd build_V(MatrixXd B, MatrixXd Wsq, MatrixXd L){
    MatrixXd SLsq = MatrixXd::Zero(n,n);
    MatrixXd V = MatrixXd::Zero(n,m);
    //Here we will compute the laplacian pseudoinverse
    EigenSolver<MatrixXd> es(L);
    for(int i = 0 ; i < n ; i ++){
        if(real(es.eigenvalues()[i]) > 1e-10){
            VectorXd evector = es.eigenvectors().col(i).real();
            double evalue = es.eigenvalues()[i].real();
            SLsq = SLsq +( 1/sqrt(evalue)*(evector*evector.transpose()));
        }
    }
    write_matrix(SLsq, "PseudoinverseSQRT.csv");
    V = SLsq*B.transpose()*Wsq;
    //cout<<V<<endl;
    write_matrix(V, "matrixV.csv");
    return V;
}

//This method returns the upper barrier to control the increment of eigenvalues of the spectral sparsifier
double upperPotential(double u, MatrixXd A){
    MatrixXd I  = MatrixXd::Identity (n,n);
    return (u*I-A).inverse().trace();
}

//This method returns the lower barrier to control the decrement of eigenvalues of the spectral sparsifier
double lowerPotential(double l, MatrixXd A){
    MatrixXd I  = MatrixXd::Identity (n,n);
    return (A-l*I).inverse().trace();
}

//The following two methos "getU" and "getL" are used in the main algorithm to find a suitable scalar t and to add tvv^T to the spectral sparsifier

double getU(VectorXd v, MatrixXd A, double u, double deltaU){
    MatrixXd I  = MatrixXd::Identity (n,n);
    MatrixXd B = ((u+deltaU)*I-A).inverse();
    double upper1 = upperPotential(u, A);
    double upper2 = upperPotential(u+deltaU, A);
    double a = v.transpose()*B*v;
    double b = v.transpose()*(B*B)*v;
    return a+(b/(upper1 - upper2));
}

double getL(VectorXd v, MatrixXd A, double l, double deltaL){
    MatrixXd I  = MatrixXd::Identity (n,n);
    MatrixXd B = (A-(l+deltaL)*I).inverse();
    double lower1 = lowerPotential(l, A);
    double lower2 = lowerPotential(l+deltaL, A);
    double a = v.transpose()*B*v;
    double b = v.transpose()*B*B*v;
    return -a+(b/(lower2 - lower1));
}

//The main algorithm

VectorXd BSS(MatrixXd V, int d){
    //Here we set the initial parameters
    MatrixXd A = MatrixXd::Zero(n,n);
    VectorXd s = VectorXd::Zero(m);
    VectorXd v = VectorXd::Zero(n);
    double t, u0, l0, ui, li, Ui, Li, eU, eL, deltaL, deltaU;
    eL = 1/sqrt(d);
    cout<<"eL = "<<eL<<endl;
    eU = (sqrt(d)-1)/(d+sqrt(d));
    cout<<"eU = "<<eU<<endl;
    l0 = -n/eL;
    cout<<"l0 = "<<l0<<endl;
    u0 = n/eU;
    cout<<"u0 = "<<u0<<endl;
    deltaL = 1;
    cout<<"deltaL = "<<deltaL<<endl;
    deltaU = (sqrt(d)+1)/(sqrt(d)-1);
    cout<<"deltaU = "<<deltaU<<endl;
    ui = u0;
    li = l0;
    int Q = d*n, i;
    int j = 0;
    
    //The main loop, here the spectral sparsifier is constructed
    for(int q = 0 ; q < Q ; q++){
        i = q%m;
        v = V.col(i);
        Ui = getU(v, A, ui, q*deltaU);
        Li = getL(v, A, li, q*deltaL);
        if(delta < Ui && Ui <= Li){
            cout<<"i = "<<i<<endl;
            cout<<"Ui = "<<Ui<<endl;
            cout<<"Li = "<<Li<<endl;
            t = 2/(Ui+Li);
            cout<<"t = "<<t<<endl;
            //cout<<"Vector = "<<v.transpose()<<endl;
            s(i) = s(i) + t;
            A = A+t*(v*v.transpose());
            //cout<<A<<endl;
            cout<<endl;
            ui = u0 + q*deltaU;
            li = l0 + q*deltaL;
        }
    }
    
    //After constructing the spectral sparsifier we write the matrix in a csv file
    write_matrix(A, "matrixA.csv");
    EigenSolver<MatrixXd> es(A);
    //cout<<es.eigenvalues()<<endl;
    double emin=real(es.eigenvalues()[0]), emax=real(es.eigenvalues()[0]);
    for(int i = 0 ; i < es.eigenvalues().size() ; i++){
        if(real(es.eigenvalues()[i]) < emin)
            emin = real(es.eigenvalues()[i]);
        if(real(es.eigenvalues()[i]) > emax)
            emax = real(es.eigenvalues()[i]);
    }
    cout<<emax<<endl;
    cout<<emin<<endl;
    cout<<"Cota = "<<(u0+Q*deltaU)/(l0+Q*deltaL)<<endl;
    //cout<<"Vector s = "<<s.transpose()<<endl;
    return s;
}

//The following three functions are utility functions

//We compare the eigenvalues to order the list of eigendata
bool compare_pairs(eigendata x, eigendata y){
    return x.first < y.first;
}


//This preprocessing helps to get the eigenvalues in sorted order with the indices
void laplacian_preprocessing(MatrixXd A, int k, string filename){
    int l;
    vector<eigendata> eigenvalues;
    eigenvalues.resize(n);
    EigenSolver<MatrixXd> es(A);
    cout<<"Eigen values of Laplacian"<<endl;
    cout<<es.eigenvalues()<<endl;
    for(int i = 0 ; i < n ; i++){
        eigenvalues[i] = make_pair(real(es.eigenvalues()[i]), i);
    }
    sort(eigenvalues.begin(), eigenvalues.end(), compare_pairs);
    write_matrix(es.eigenvectors().real(), "eigenvectores_"+filename);
    MatrixXd V(n, k);
    for (int j = 0 ; j < k ; j ++){
        l = eigenvalues[j].second;
        for(int i = 0 ; i < n; i++){
            V(i,j) = real(es.eigenvectors()(i,l));
        }
    }
    cout<<V<<"\n"<<endl;
    write_matrix(V, filename);
    
}


//This method read the points to cluster from an standard input
void read_data(){
    ofstream myfile;
    myfile.open ("puntos_originales.csv");//En este archivo se almacenaran los puntos leidos desde un archivo de texto plano
    cin >> n >> d >> epsilon >> delta;//aqui empezamos a leer los puntos desde un archivo de texto plano
    cout<<epsilon<<endl;
    m = n*(n-1)/2;
    double u,v;
    for(int i = 0 ; i < n ; i++){
        cin >> u >> v;
        VectorXd x(2);
        x(0) = u;
        x(1) = v;
        myfile <<u<<","<<v<<"\n";
        //cout<<x<<endl;
        puntos.push_back(x);//almacenamos los puntos en una lista
    }
    myfile.close();
}

//The main functions, where we execute tests

int main(){
    read_data();
    int d = (int) 1/pow(epsilon,2);
    cout<<d<<endl;
    MatrixXd A = build_Adjacency_matrix();
    MatrixXd B = build_B();
    write_matrix(B, "matrixB.csv");
    //cout<<B<<endl;
    MatrixXd W = build_W();
    write_matrix(W, "matrixW.csv");
    //cout<<W<<endl;
    MatrixXd Wsq = build_Wsq(W);
    //cout<<Wsq<<endl;
    MatrixXd D = build_D();
    MatrixXd L = D-A;
    write_matrix(L, "laplacian.csv");
    MatrixXd Dsq = build_Dsq(D);
    //cout<<D<<endl;
    MatrixXd NL = build_NormalizedLaplacian(B, W, Dsq);
    write_matrix(NL, "NormalizedLaplacian.csv");
    //cout<<NL<<endl;
    MatrixXd V = build_V(B, W, NL);
    //cout<<V<<endl;
    VectorXd s = BSS(V, d); 
    MatrixXd W1 = build_reweightedW(W, s);
    //write_matrix(W1, "reweightedW.csv");
    //MatrixXd S = s.asDiagonal();
    //cout<<"Matrix S\n"<<S<<endl;
    //write_matrix(S, "matrixS.csv");
    MatrixXd Lh = B.transpose()*W1*B;
    write_matrix(Lh, "laplacian_sparsifiered.csv");
    //cout<<L<<endl;
    //cout<<Lh<<endl;
    laplacian_preprocessing(NL, 2, "points.csv");
    laplacian_preprocessing(Lh, 2, "points_sparsifiered.csv");
}
