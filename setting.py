import os

if 'VEHelper_Workpath' in os.environ:
    MAIN_WORK_PATH = os.environ['VEHelper_Workpath']
else:
    print ('''ERROR : Failed to get ENV VARIABLE "VEHelper_Workpath"''')

frontend_work_path = MAIN_WORK_PATH + 'djangoService\\vehelper\\'
backend_work_path = MAIN_WORK_PATH

web_pdassigner_title = 'Service - PD to Team'
web_pdassigner_repdtest_row_template = 'TCID\t\tTCTITLE\t\tTCTITLE\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tTEAM\n'


