pipeline {
  agent any
  stages {
    stage('Set Environment') {
      steps {
        sh 'export LD_LIBRARY_PATH=/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:${LD_LIBRARY_PATH}'
        echo 'Add path for Math Kernal Libary'
      }
    }
    stage('Build program') {
      steps {
        sh 'gcc -march=native -fprofile-arcs -ftest-coverage -O3 -unroll-aggressive -o main main.c Pgen.c lte_dfts.c -fPIC -DMKL_ILP64 -m64 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_intel_ilp64 -lmkl_gf_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -mavx2'
        echo 'Build main'
      }
    }
    stage('Run performance measuremance') {
      steps {
        sh '''export LD_LIBRARY_PATH=/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:${LD_LIBRARY_PATH}
./main'''
      }
    }
    stage('Test code coverage') {
      parallel {
        stage('Test code coverage') {
          steps {
            sh 'gcovr --branches --xml-pretty -r . > gcovr.xml'
            cobertura(coberturaReportFile: '**/gcovr.xml')
          }
        }
        stage('Email Notification') {
          steps {
            emailext(subject: 'CI build notification', attachLog: true, replyTo: 'danghoaison1991@gmail.com', to: 'danghoaison1991@gmail.com', body: 'This mail report CI build')
          }
        }
      }
    }
    stage('Clear CI') {
      steps {
        sh 'rm -rf main'
      }
    }
    stage('Done') {
      steps {
        cleanWs(cleanWhenFailure: true)
      }
    }
  }
}