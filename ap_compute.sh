output_file=$2
input_file=$1
matlab_dir=/afs/cs.stanford.edu/u/rwitten/scratch/VOC2007/VOCdevkit/

function name()
{
    cwd=`pwd`
    cd `dirname $1`
    echo `pwd`/`basename $1`
    cd $cwd
}


full_name=`name $input_file`
matlab -nosplash -nojvm -r "cd $matlab_dir; spl_compute_pr('$full_name'); exit;" | grep magic  > $output_file
