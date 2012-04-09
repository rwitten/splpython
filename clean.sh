find . -name '*swp' | xargs rm -f 
find . -name '*swo' | xargs rm -f
find . -name '*swn' | xargs rm -f
find . -name '*pyc' | xargs rm -f 

rm -r output/*
