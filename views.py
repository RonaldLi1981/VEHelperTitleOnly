#from django.shortcuts import render
# _*_ encoding:utf-8 _*_
# Create your views here.

from django.http import HttpResponse, StreamingHttpResponse
#import cgi, cgitb
import xlsxwriter
import socket
import sys
import os
import io
import re
if 'VEHelper_Workpath' in os.environ:
    sys.path.append( os.environ['VEHelper_Workpath'] + 'djangoService\\vehelper\\pdassigner')
    sys.path.append(os.environ['VEHelper_Workpath'])
import page_template as p
from django.views.decorators.csrf import csrf_exempt
from setting import web_pdassigner_repdtest_row_template as row_template
from setting import frontend_work_path as work_path

i = 1
dst_ip_port = ('127.0.0.1', 42504)


def index(request):
 
    output_page = '<html>'
    output_page += '<head>'
    output_page += '<meta http-equiv="Content-Type" Content="text/html; Charset=gb2312">'
    output_page += '''<link rel="shortcut icon" href="/static/favicon.ico"/>'''
    output_page += '<title>PD Assigner'
    output_page += '</title>\n'
    output_page += '</head>'
    output_page += '<body>\n'
    output_page += p.page.page_body_b
    output_page += '</body>\n'
    output_page += '</html>'

    return HttpResponse(output_page)

@csrf_exempt
def proceedPdTitles(request):
    global i
    data = ''
    pdtitles = []
    repdtest = False
    export_filename = 'REPDTEST_RQM.xlsx'
    
    if request.POST:
        if request.POST['output_format'] == 'REPDTEST':
            repdtest = True
        data = request.POST['pdtitles']
        if len(data)>1:
            pdtitles = data
            with socket.socket() as s:
                print ('Sending REQUEST - {}'.format(i))
                i += 1
                s.connect(dst_ip_port)
                requestName = 'TitleToTeam'
                s.sendall((requestName + ' ' + pdtitles).encode())
                print ('SENT')
#                print (requestName + ' ' + pdtitles[0])
                result = s.recv(1000000)
                print ('RECEIVED')
                result_str = result.decode()
                s.close()
            
            if (repdtest):
                result_list = result_str.split('\r\n')
                result_teams = []
                result_pdtitles = []
                for r in result_list:
                    if len(r.split('\t')) >1:
                        result_teams.append(r.split('\t')[0])
                        result_pdtitles.append(r.split('\t')[1])
                
                result_str = ''
                # Generate EXCEL file for RQM importing
                export_io = io.BytesIO()
                xlsx_f = xlsxwriter.Workbook(export_io, {'in_memory': True})
                style_title= xlsx_f.add_format({'bold':True, 'bg_color':'#44AACC', 'top':4, 'left':4, 'right':4, 'bottom':4})
                style_text= xlsx_f.add_format({'bg_color':'#99CCDD', 'top':2, 'left':2, 'right':2, 'bottom':2})
                sht1 = xlsx_f.add_worksheet('Test Cases')
                sht1.set_column('A:A', 32)
                sht1.set_column('B:B', 100)
                sht1.set_column('C:C', 10)
                sht1.write(0,0, 'TP Name', style_title)
                sht1.write(0,1, 'Owner', style_title)
                sht1.write(1,0, request.POST['release_name'] + ' REPDTEST TP', style_text)
                sht1.write(1,1, 'fliu', style_text)
                sht1.write(2,0, 'Test Case ID', style_title)
                sht1.write(2,1, 'Test Case Headline', style_title)
                sht1.write(2,2, 'Team', style_title)
                i_cur_row = 3
                for team, pdtitle in zip( result_teams, result_pdtitles ):
                    tmp = re.match('^\D*(\d*).*$', pdtitle[0:15] )
                    if tmp != None:
                        pd_number = tmp.group(1)
                    else:
                        pd_number = 'FAILED_TO_GET_PD_NUMBER'
                    sht1.write(i_cur_row, 0, request.POST['release_name'] + '_PD_' + pd_number, style_text)
                    sht1.write(i_cur_row, 1, pdtitle, style_text)
                    sht1.write(i_cur_row, 2, team, style_text)
                    i_cur_row += 1
                    row = row_template
                    row = row.replace('TCID', request.POST['release_name'] + '_PD_' + pd_number)
                    row = row.replace('TEAM', team)
                    row = row.replace('TCTITLE', pdtitle)
                    result_str += row
                xlsx_f.close()
                export_io.seek(0)
                # response = StreamingHttpResponse(file_iterator(export_filepath + '\\' + export_filename))
                response = StreamingHttpResponse(export_io)
                response['Content-Type'] = 'application/octet-stream'
                response['Content-Disposition'] = 'attachment;filename="{0}"'.format(export_filename)
                return response
            else:                
                output_page = '<Meta http-equiv="Content-Type" Content="text/html; Charset=gb2312">'
                output_page += '\n'
                output_page += '<html>'
                output_page += '''<head><link rel="shortcut icon" href=" /favicon.ico" />'''
                output_page += '<title>PD Auto Assigner - Results'
                output_page += '</title>\n'
                output_page += '</head>'
                output_page += '<body>\n'
                output_page += '<textarea cols="150" rows="40" style="background-color:90EE90">' + result_str + '</textarea>' #overflow-x:hidden;
                output_page += '</body>\n'
                output_page += '</html>'

                return HttpResponse(output_page)
        else:
            return HttpResponse('Empty input, please input your PD Titles and submit again!')
    

def file_iterator(filename, chunk_size=512):
    with open(filename, mode='r') as f: #, encoding='UTF-8'
        while True:
            output = f.read(chunk_size)
            if output:
                yield output
            else:
                break

def getRqmImportingCfg(request):
    response = StreamingHttpResponse(file_iterator('file/RQM Importing - REPDTEST.Cfg'))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format('RQM_Importing_File.Cfg')
    return response

