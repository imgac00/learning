#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  aaaaatupian.py
#  

#
#
import os
import requests
from threading import Thread
from lxml import etree
from gui_tupian import GuiTupia

ee=['4kfengjing', '4kmeinv', '4kyouxi', '4kdongman', '4kyingshi', '4kmingxing',
'4kqiche', '4krenwu', '4kbeijing', '4kdongwu', '4kmeishi', '4kzongjiao']
print("输入爬取类型（前面数字代号）\n0:'4kfengjing'; 1:'4kmeinv'; \
2:'4kyouxi'; 3:'4kdongman'; \n4:'4kyingshi' ; 5:'4kmingxing';  6:'4kqiche'; \
7:'4krenwu'; \n8:'4kbeijing' ; 9:'4kdongwu'; 10:'4kmeishi'; 11:'4kzongjiao'\n")
jun=ee[int(input())]
iii=GuiTupia(jun)
dd=iii.get_hdurl()
nin=[]
path = ('%s' % jun)
if path not in os.listdir('E:\\'):
        os.mkdir('E:\\%s' % path)

def DownloadTu(url):
        
        url_d='http://pic.netbian.com/tupian/'+str(url[0][:])+'.html'
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) \
        Chrome/65.0.3314.0 Safari/537.36 SE 2.X MetaSr 1.0'}
        r=requests.get(url_d, headers=headers)
        html=etree.HTML(r.content, parser=etree.HTMLParser(encoding='GBK'))

        src_list=html.xpath('//img/@src')
        jj=html.xpath('//div[@class="photo-hd"]//text()')
	# ~ print(src_list)

        for src in src_list[:1]:
		# ~ dd='\d{2}'
		# ~ ids=re.findall(dd,src)
		
                urls='http://pic.netbian.com'+str(src)
		# ~ print(len(urls))
                content=requests.get(urls,headers=headers).content
                for j in jj:
                        aaa=j

                filename="E:\\{}\\{}.jpg".format(jun, aaa)
                print("save:{}".format(filename))
                with open(filename, "wb") as f:
                        f.write(content)
                f.close()

def main():
        for url in dd:
                xie=Thread(target=DownloadTu, args=(url,))
                nin.append(xie)
                xie.start()
if __name__=='__main__':
        main()
        for ii in nin:
                ii.join()
        print('下载完成')
