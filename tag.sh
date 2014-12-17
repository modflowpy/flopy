for t in `git tag`
do
    git tag -d $t
    git push origin :$t
done