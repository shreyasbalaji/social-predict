#!/bin/bash
cd /Users/vanshikajain/Downloads/MIT_Sophomore/hackmit/
instaloader "#eeifshemaisrael"
cd \#eeifshemaisrael/
for g in *.txt
do 
	rm "$g"
done
for f in *.xz
do
	xz --decompress "$f"
done
