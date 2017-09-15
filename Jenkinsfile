#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// -- HELPER AUXILLARY FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Construct the relative path of the build directory
String build_directory_rel( String build_config )
{
  if( build_config.equalsIgnoreCase( 'release' ) )
  {
    return "build/release"
  }
  else
  {
    return "build/debug"
  }
}

////////////////////////////////////////////////////////////////////////
// -- FUNCTIONS RELATED TO BUILD
// This encapsulates running of unit tests
def docker_build_image( String src_dir_abs )
{
  String project = "hiptensorflow"
  String build_type_name = "build-ubuntu-16.04"
  String dockerfile_name = "dockerfile-${build_type_name}"
  String build_image_name = "${build_type_name}"
  def build_image = null

  stage('ubuntu-16.04 image')
  {
    dir("${src_dir_abs}/hiptensorflow/docker")
    {
      def user_uid = sh( script: 'id -u', returnStdout: true ).trim()
      build_image = docker.build( "${project}/${build_image_name}:latest", "-f ${dockerfile_name} --build-arg user_uid=${user_uid} ." )
    }
  }

  return build_image
}

////////////////////////////////////////////////////////////////////////
// Checkout the desired source code and update the version number
String checkout_and_version( String workspace_dir_abs )
{
  String source_dir_abs = "${workspace_dir_abs}/src/"

  dir("${workspace_dir_abs}")
  {
    stage("github clone")
    {
      deleteDir( )

      dir( "${source_dir_abs}/hiptensorflow" )
      {
        // checkout hiptensorflow
        checkout([
            $class: 'GitSCM',
            branches: scm.branches,
            doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
            extensions: scm.extensions + [[$class: 'CloneOption', depth: 1, noTags: false, reference: '', shallow: true]],
            submoduleCfg: [],
            userRemoteConfigs: scm.userRemoteConfigs
            ])
      }

      // checkout hipeigen
      checkout( [$class: 'GitSCM',
          branches: [[name: '*/develop']],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'CloneOption', depth: 1, noTags: false, reference: '', shallow: true], [$class: 'RelativeTargetDirectory', relativeTargetDir: "${source_dir_abs}/hipeigen"]],
          submoduleCfg: [],
          userRemoteConfigs: [[credentialsId: '0a2d23e5-f8c9-4d45-abdf-1124fd394584', url: 'https://github.com/ROCmSoftwarePlatform/hipeigen.git']]
          ])


      // checkout hcBLAS
      checkout( [$class: 'GitSCM',
          branches: [[name: '*/master']],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'CloneOption', depth: 1, noTags: false, reference: '', shallow: true], [$class: 'RelativeTargetDirectory', relativeTargetDir: "${source_dir_abs}/hcBLAS"]],
          submoduleCfg: [],
          userRemoteConfigs: [[credentialsId: '0a2d23e5-f8c9-4d45-abdf-1124fd394584', url: 'https://github.com/ROCmSoftwarePlatform/hcBLAS.git']]
          ])

      // checkout hcFFT
      checkout( [$class: 'GitSCM',
          branches: [[name: '*/master']],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'CloneOption', depth: 1, noTags: false, reference: '', shallow: true], [$class: 'RelativeTargetDirectory', relativeTargetDir: "${source_dir_abs}/hcFFT"]],
          submoduleCfg: [],
          userRemoteConfigs: [[credentialsId: '0a2d23e5-f8c9-4d45-abdf-1124fd394584', url: 'https://github.com/ROCmSoftwarePlatform/hcFFT.git']]
          ])

      // checkout hcRNG
      checkout( [$class: 'GitSCM',
          branches: [[name: '*/master']],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'CloneOption', depth: 1, noTags: false, reference: '', shallow: true], [$class: 'RelativeTargetDirectory', relativeTargetDir: "${source_dir_abs}/hcRNG"]],
          submoduleCfg: [],
          userRemoteConfigs: [[credentialsId: '0a2d23e5-f8c9-4d45-abdf-1124fd394584', url: 'https://github.com/ROCmSoftwarePlatform/hcRNG.git']]
          ])

      // checkout MiOpen
      checkout( [$class: 'GitSCM',
          branches: [[name: '*/develop']],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'CloneOption', depth: 1, noTags: false, reference: '', shallow: true], [$class: 'RelativeTargetDirectory', relativeTargetDir: "${source_dir_abs}/MLOpen"]],
          submoduleCfg: [],
          userRemoteConfigs: [[credentialsId: '0a2d23e5-f8c9-4d45-abdf-1124fd394584', url: 'https://github.com/AMDComputeLibraries/MLOpen.git']]
          ])
    }
  }

  return source_dir_abs
}


////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure, build and package commands
// Leverages docker containers to encapsulate the build in a fixed environment
def docker_build_inside_image( def build_image, String build_config, String src_dir_abs, String build_dir_abs )
{
  build_image.inside( )
  {
    stage("build ${build_config}")
    {
      String install_prefix = "/opt/rocm/hiptensorflow"

      withEnv(["PATH=${PATH}:/opt/rocm/bin:/opt/rocm/hip/bin", "HIP_EIGEN_PATH=${src_dir_abs}/hipeigen"])
      {
        sh  """#!/usr/bin/env bash
            set -x
            cd ${src_dir_abs}/hcBLAS && bash -x ./build.sh && sudo dpkg -i build/*.deb
            cd ${src_dir_abs}/hcRNG && bash -x ./build.sh && sudo dpkg -i build/*.deb
            cd ${src_dir_abs}/hcFFT && bash -x ./build.sh && sudo dpkg -i build/*.deb
            cd ${src_dir_abs}/MLOpen && mkdir -p build && cd build
            CXX=/opt/rocm/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc" ..
            sudo make -j \$(nproc) install
            cd ${src_dir_abs}/hiptensorflow
            tensorflow/tools/ci_build/builds/configured gpu --disable-gcp
            bazel test -c opt --config=cuda //tensorflow/...
            bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow-hip
          """
      }
    }

    // stage("packaging")
    // {
    //   sh "cd ${build_dir_abs}/library-build; make package"
    //   archiveArtifacts artifacts: "${build_dir_rel}/library-build/*.deb", fingerprint: true
    //   archiveArtifacts artifacts: "${build_dir_rel}/library-build/*.rpm", fingerprint: true
    //   sh "sudo dpkg -c ${build_dir_abs}/library-build/*.deb"
    // }
  }

  return void
}

////////////////////////////////////////////////////////////////////////
// This encapsulates running of unit tests
def docker_upload_artifactory( String build_config, String workspace_dir_abs )
{
  def rocblas_install_image = null
  String image_name = "hiptensorflow-ubuntu-16.04"
  String artifactory_org = "${env.JOB_NAME}".toLowerCase( )

  // def ( String build_dir_rel, String build_dir_abs ) = build_directory( build_config, workspace_dir_abs )
  String build_dir_rel = build_directory_rel( build_config );
  String build_dir_abs = "${workspace_dir_abs}/" + build_dir_rel

  stage( 'artifactory' )
  {
    dir( "${build_dir_abs}/docker" )
    {
      //  We copy the docker files into the bin directory where the .deb lives so that it's a clean
      //  build everytime
      sh "cp -r ${workspace_dir_abs}/docker/* .; cp ${build_dir_abs}/library-build/*.deb ."
      rocblas_install_image = docker.build( "${artifactory_org}/${image_name}:${env.BUILD_NUMBER}", "-f dockerfile-${image_name} ." )
    }

    // docker.withRegistry('http://compute-artifactory:5001', 'artifactory-cred' )
    // {
    //  rocblas_install_image.push( "${env.BUILD_NUMBER}" )
    //  rocblas_install_image.push( 'latest' )
    // }

    // Lots of images with tags are created above; no apparent way to delete images:tags with docker global variable
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${artifactory_org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }
}

////////////////////////////////////////////////////////////////////////
// This routines defines the pipeline of the build; the order that various helper functions
// are called.
// Calls helper routines to do the work and stitches them together
def hiptensorflow_build_pipeline( String build_config )
{
  // Convenience variables for common paths used in building
  String workspace_dir_abs = pwd()

  // Checkout all dependencies
  String source_dir_abs = checkout_and_version( "${workspace_dir_abs}" )

  // Create/reuse a docker image that represents the hiptensorflow build environment
  def hiptensorflow_build_image = docker_build_image( "${source_dir_abs}" )

  String build_dir_rel = build_directory_rel( build_config );
  String build_dir_abs = "${workspace_dir_abs}/" + build_dir_rel

  // Build hiptensorflow inside of the build environment
  docker_build_inside_image( hiptensorflow_build_image, "${build_config}", "${source_dir_abs}", "${build_dir_abs}" )

  // docker_upload_artifactory( "${build_config}", "${workspace_dir_abs}" )

  return void
}

////////////////////////////////////////////////////////////////////////
// -- MAIN
// This following are build nodes; start of build pipeline
node('docker && rocmtest')
{
  hiptensorflow_build_pipeline( 'Release' )
}
