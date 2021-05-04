#!/usr/bin/lua

local function measureFile(n)
 local o = io.open(n, "rb")
 local v = o:seek("end")
 o:close()
 return v
end

local function addCase(i, o)
 local samplesI = measureFile(i) // 8
 local samplesO = measureFile(o) // 8
 local samplesIX = math.min(samplesI * 2, samplesO) // 2
 local samplesOX = samplesIX * 2
 os.execute("head -c " .. (samplesIX * 8) .. " " .. i .. " >> dataset/i.raw")
 os.execute("head -c " .. (samplesOX * 8) .. " " .. o .. " >> dataset/o.raw")
end

local i = 0
while true do
 local f = io.open("dataset/index/" .. i, "r")
 if not f then break end
 while true do
  local l = f:read()
  if not l then break end
  if l ~= "" then
   print(l)
   os.execute("ffmpeg -i \"" .. l .. "\" /media/ramdisk/tmp.wav")
   os.execute("sox /media/ramdisk/tmp.wav -t f32 --endian little -c 2 -r 24000 /media/ramdisk/tmp.o.raw")
   os.execute("sox /media/ramdisk/tmp.wav -t f32 --endian little -c 2 -r 12000 /media/ramdisk/tmp.i.raw")
   os.execute("sox /media/ramdisk/tmp.wav -t f32 --endian little -c 2 -r 6000 /media/ramdisk/tmp.x.raw")
   os.execute("rm /media/ramdisk/tmp.wav")
   addCase("/media/ramdisk/tmp.x.raw", "/media/ramdisk/tmp.i.raw")
   addCase("/media/ramdisk/tmp.i.raw", "/media/ramdisk/tmp.o.raw")
   os.execute("rm /media/ramdisk/tmp.o.raw")
   os.execute("rm /media/ramdisk/tmp.i.raw")
   os.execute("rm /media/ramdisk/tmp.x.raw")
  end
 end
 f:close()
 os.execute("mv dataset/index/" .. i .. " dataset/index-done/")
 i = i + 1
end

