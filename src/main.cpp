#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>

#include <mlpack/core.hpp>
#include "mlpack/core/data/load.hpp"
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/mean_shift/mean_shift.hpp>

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sys/dir.h>

Eigen::MatrixXd V;
Eigen::MatrixXi F;

using namespace arma;
using namespace std;
using namespace mlpack::kmeans;
using namespace mlpack::dbscan;
using namespace mlpack::meanshift;
using namespace Eigen;
using namespace cv;

vector<string> listFile(char folder_name[])
{
    DIR *pDIR;
    struct dirent *entry;
    vector<string> files;
    if( pDIR=opendir(folder_name) ){
            while(entry = readdir(pDIR)){
                    if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
                   // cout << entry->d_name << "\n";

                    files.push_back(entry->d_name);
                  //  string folder_name = "Michelle_pic/";
            }
            closedir(pDIR);
            std::sort(files.begin(), files.end() );
    }
    return files;
}

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

void removeRow_I(Eigen::MatrixXi& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}



int main()
{

  string directory = "potato_ply_mesh_1408/N/S/";
 // string directory = "potato_ld_mesh/";
 // string directory = "small_file_size/";
  char *shapes= &directory[0u];

  ofstream file, file_eye;
  file.open ("Eye_analysis_N_revised.txt", ios::app);
//  file_eye.open ("Potato_eyes_validate.txt", ios::app);

  vector<string> cloud_names = listFile(shapes);

  for (int i = 0; i < cloud_names.size(); i++)   //38
  {

    string filename = directory + cloud_names[i];
    // Load a mesh in OFF format
    igl::readPLY(filename, V, F);
    cout<<cloud_names[i]<<"; index: "<<i<<endl;
    // Alternative discrete mean curvature


  /*  VectorXd X;
    igl::doublearea(V, F, X);

    float surface_area = 0;
    for (int i = 0; i < X.size(); i++)
    {
        surface_area += X(i);
    }
  //  cout<<surface_area<<endl;*/


    MatrixXd HN;
    SparseMatrix<double> L,M,Minv;
    igl::cotmatrix(V,F,L);
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
    igl::invert_diag(M,Minv);
    // Laplace-Beltrami of position
    HN = -Minv*(L*V);
    // Extract magnitude as mean curvature
    VectorXd H = HN.rowwise().norm();



    // Compute curvature directions via quadric fitting
    MatrixXd PD1,PD2;
    VectorXd PV1,PV2;
    igl::principal_curvature(V,F,PD1,PD2,PV1,PV2);

    H = 0.5*(PV1+PV2);

    MatrixXd C(V.rows(), 3);        //define mesh colour
    C = C.setOnes();


    vector<float> z_values;
    for (int s = 0; s < V.rows(); s++)
    {
        z_values.push_back(V(s,2));
    }

    float min_value = *std::min_element(z_values.begin(), z_values.end());
    float max_value = *std::max_element(z_values.begin(), z_values.end());
    float thold = min_value + (max_value - min_value)/6;
   // float thold = max_value - (max_value - min_value)/6;


    vector<int> index;
    vector<float> curs;
    for(int i = 0; i < V.rows(); i++)
    {
        //myfile<<V(i,0)<<","<<V(i,1)<<","<<V(i,2)<<endl;
        if (H(i) < -1.8 && V(i,2) > thold)                         //-0.8
        {
            index.push_back(i);
            curs.push_back(H(i));
        }
    }


    mat V_reduced_temp;
    mat V_reduced(index.size(), 3);


    for (int i = 0; i < V_reduced.n_rows; i++)
    {
        V_reduced(i,0) =  V(index[i], 0);
        V_reduced(i,1) =  V(index[i], 1);
        V_reduced(i,2) =  V(index[i], 2);
    }

 // cout<<curs.size()<<" "<<V_reduced.n_rows<<endl;


   size_t clusters;
   arma::Row<size_t> assignments;
   arma::mat centroids;
   bool forceConvergence = true;
   MeanShift<> m;


 /*  m.Radius(0.15);
   m.Cluster((arma::mat)V_reduced.t(), assignments, centroids, forceConvergence);
   assignments = assignments + 1;*/

   DBSCAN<> d(0.1, 20, true);                                       // 0.1,  20
   d.Cluster((arma::mat)V_reduced.t(), assignments, centroids);
   assignments = assignments + 1;

   MatrixXd clusterLabel(index.size(), 1);
   vector<int> labels(index.size());

   int cluster_num = 0;

 //  file_eye<<cloud_names[i]<<endl;

   for (int i = 1; i < assignments.max()+1; i++)
   {
       int tmp = 0;
       vector<int> idx;
       for (int j = 0; j < index.size(); j++)
       {
            if (assignments(j) == i)
            {
                tmp++;
                idx.push_back(j);
            }
       }

       if(tmp < 40)                                                              //20
       {
           for(int m = 0; m < idx.size(); m++)
           {
               assignments(idx[m]) = 0;
               clusterLabel(idx[m],0) = assignments(idx[m]);
               labels[idx[m]] = assignments(idx[m]);
           }
       }
       else
       {
           cluster_num++;                                       // final group number
           for(int n = 0; n < idx.size(); n++)
           {
               clusterLabel(idx[n],0) = assignments(idx[n]);         // the length of clusterLabel should be same with assignments, use index to project to 2D.
               labels[idx[n]] = assignments(idx[n]);
            //   cout<<assignments(idx[n])<<endl;
           }

       }
   }

   int new_label = 0;
   for (int d = 1; d < labels.size()+1; d++)
   {
       if (std::find(labels.begin(), labels.end(), d) != labels.end())
       {
           new_label++;
           for (int e = 0; e < labels.size(); e++)
           {
                if (labels[e] == d)
                {
                    labels[e] = new_label;
                }
           }
       }
   }


   MatrixXd label_final(index.size(), 1);
   for (int size=0; size<labels.size(); size++)
   {
       label_final(size, 0) = labels[size];
  //     cout<<label_final(size,0)<<",  ";
   }


   vector<float> ave_temp_vec;
   vector<float> each_cur;
   float ave_temp = 0;
   int ver_num;
   MatrixXd ave_cur (labels.size(), 2);

   for (int i = 1; i <= new_label; i++)
   {
       ver_num = 0;
       for (int j = 0; j < label_final.rows(); j++)
       {
           if (label_final(j,0) == i)
           {
               ave_temp_vec.push_back(curs[j]);
               ver_num++;
           }
       }
       sort(ave_temp_vec.begin(), ave_temp_vec.end());
       for (int m = 0; m < (int)(ave_temp_vec.size()*0.3); m++)    //*0.3
       {
           ave_temp += ave_temp_vec[m];
       }
       ave_temp = ave_temp / (int)(ave_temp_vec.size()*0.3);        //*0.3
       each_cur.push_back(ave_temp);
       ave_temp_vec.clear();
    //   cout<<"eye_num: "<<i<<" ave cur: "<<ave_temp<<endl;
   //    file_eye<<i<<","<<ave_temp<<","<<ver_num<<endl;
   }

   float sum_curvature = 0;
   for (int aa = 0; aa < each_cur.size(); aa++)
   {
       sum_curvature += each_cur[aa];
   }
   sort(each_cur.begin(), each_cur.end());
   float ave_curvature = sum_curvature / each_cur.size() * (-0.902) - 0.111;
   float max_cur = each_cur[0] * (-1.034) + 0.05;

   cout<<"Eye number: "<<new_label<<" average curvature : "<<ave_curvature<<";  Max_cur: "<<max_cur<<endl;


  file<<cloud_names[i]<<","<<new_label<<","<<ave_curvature<<","<<max_cur<<endl;


   MatrixXd C_cluster(V_reduced.n_rows, 3);
  // igl::jet(label_final, true, C_cluster);

   for (int i = 0; i < V.rows(); i++)
   {
       C(i, 0) = 120;
       C(i, 1) = 120;
       C(i, 2) = 0;
   }


 for (int c = 1; c < cluster_num+1; c++)
 {

   for (int i = 0; i < index.size(); i++)
   {
       if (label_final(i,0) == c)
       {
       C(index[i], 0) = 120;
       C(index[i], 1) = 0;
       C(index[i], 2) = 0;
       }
   }

  /*     igl::opengl::glfw::Viewer viewer;
       viewer.data().set_mesh(V, F);
       viewer.data().set_colors(C);
       viewer.launch();*/
 }

  //  cout<<cluster_num<<endl;


 /*   igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    viewer.launch();*/
   }
  // file.close();

}
