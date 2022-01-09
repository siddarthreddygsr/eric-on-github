
## produced by ERIC

this directory consists of some mid files produced by ERIC to make it easily accissable on every platform some files have been converted to mp3 file format using ffmpeg and timidity with the following command 

```bash
 timidity test_output-epoch.mid   -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k run-3-epoch-151.mp3
```
mp3 files can be played conviniently with mpv using 
```bash
 mpv filename.mp3
 ```
or any other music player of choice. Similarly mid files can be played using the following comming

```bash
 timidity filename.mid
```

ignore the naming culture of the output files 😊
