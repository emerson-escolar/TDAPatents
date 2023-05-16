#!/bin/bash


VERSION="0"

while getopts v: option
do
	case "${option}"
		in
		v) VERSION="$OPTARG";;
  esac
done

if [ $VERSION == "0" ]; then
	echo "Usage: git-release -v <version>"
	exit 1
fi

git submodule init
git submodule update

zip -r "TDAPatents-v$VERSION.zip" 180901_csv 200110_csv mappertools tdapatents LICENSE pytest.ini README.md README_Jaffe_measures.md requirements.txt VERSION
