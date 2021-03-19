#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  aaaaatupian.py
#  

#  
import os,requests,re
from bs4 import BeautifulSoup
from lxml import etree
from tkinter import *



class GuiTupia():
	def __init__(self, namedd):
		self.namedd=namedd
		
	def get_hdurl(self):
		
		aa=[]
		print('开始爬取的页面（输入数字） : ')
		ye1=int(input())
		print('结束爬取的页面（输入数字） : ')
		ye2=int(input())
		for y in range(ye1, ye2):
			if y==0:
				url='http://pic.netbian.com/'+self.namedd+'/index.html'
				
			else:
				url='http://pic.netbian.com/'+self.namedd+'/index_'+str(y)+'.html'
			print(url)
			headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3314.0 Safari/537.36 SE 2.X MetaSr 1.0'}
			r=requests.get(url,headers=headers)
			soup=BeautifulSoup(r.content, 'lxml')
				#
					
			data=soup.select('#main>div.slist>ul>li>a')
				# ~ print(data)
				#main > div.slist > ul > li:nth-child(20) > a
				
					

			for item in data:
				# ~ aa=item.find_all('a')
				# ~ print(aa[0].string)
				dd=re.findall('\d+', item.get("href"))
					# ~ result={
						# ~ 'title':item.get_text(),
						# ~ 'link':item.get('href'),
						# ~ 'imgs':item.get('img')
					# ~ }
				aa.append(dd)
				
			if len(aa)==0:
				print('超出页面数！')
			# ~ else: 
				# ~ for i in aa[:]:
					
					# ~ urls='http://pic.netbian.com/tupian/'+str(i[0][:])+'.html'
					# ~ print(urls)
				
		return aa
