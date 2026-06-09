#rsync 同步repo
rsync -avzP /d/project/DEFOM-Stereo me241123@100.70.243.113:~/stereo/DEFOM-Stereo

#scp 复制文件
scp file me241123@100.70.243.113:~/stereo/DEFOM-Stereo

#后台不中断下载
nohup wget --no-proxy -c -t 0 --timeout=30 --waitretry=5 https://www.xxx.7z > wget.log 2>&1 &   

#循环解压文件夹内所有zip
for f in *.zip; do unzip -o "$f"; done