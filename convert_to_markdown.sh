#!/bin/bash

convert_to_markdown() {

filename=$(basename -- "$1" .ipynb)
git rm -r -f "${filename}_files"
rm -r -f "${filename}_files"
echo "Removed ${filename}_files"
jupyter nbconvert --to markdown "$1"


}

convert_to_markdown "$1" 
