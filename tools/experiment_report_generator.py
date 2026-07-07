import os
import glob
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from tqdm import tqdm
def parse_scripts(script_dir):
    """解析shell脚本，提取日期、命令和结果目录"""
    experiments = {}
    for script in glob.glob(os.path.join(script_dir, "*.sh")):
        with open(script, 'r') as f:
            content = f.read()
            
            # 提取日期
            date_line = next(line for line in content.splitlines() if line.find("Date=") != -1)
            date_str = date_line.split("=")[1].strip()
            date = datetime.strptime(date_str, "%Y_%b_%d")
            
            # 提取所有torchrun命令（包括多行命令）
            commands = []
            current_command = ""
            in_command = False
            
            # 提取结果目录（包括注释掉的）
            save_dirs = []
            
            # 按顺序解析命令和结果目录
            for line in content.splitlines():
                line = line.strip()
                
                # 处理结果目录
                if line.find("save_dir=") != -1:
                    # 替换${Date}变量
                    dir_path = line.split("save_dir=")[1].strip()
                    dir_path = dir_path.replace("${Date}", date_str)
                    if dir_path.endswith("/"):
                        dir_path = dir_path[:-1]
                    save_dirs.append(dir_path)
                
                # 处理命令
                if line.find("torchrun") != -1:
                    in_command = True
                    current_command = line
                elif in_command and line.find("\\") != -1:
                    current_command += " " + line.rstrip("\\")
                elif in_command:
                    current_command += " " + line
                    commands.append(current_command)
                    in_command = False
                    current_command = ""
            

            experiments[date] = {
                'commands': commands,  # 保持为列表
                'results_dirs': save_dirs
            }
    return experiments

def generate_mrc_views(mrc_path):
    """生成mrc文件的三视图"""
    with mrcfile.open(mrc_path) as mrc:
        data = mrc.data
        mid_z = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_x = data.shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY平面
        axes[0].imshow(data[mid_z, :, :], cmap='gray')
        axes[0].set_title('XY Plane')
        
        # XZ平面
        axes[1].imshow(data[:, mid_y, :], cmap='gray')
        axes[1].set_title('XZ Plane')
        
        # YZ平面
        axes[2].imshow(data[:, :, mid_x], cmap='gray')
        axes[2].set_title('YZ Plane')
        
        plt.tight_layout()
        return fig

def create_pdf_report(date, experiment, output_dir):
    """创建单个日期的PDF报告"""
    pdf_path = os.path.join(output_dir, f"experiment_report_{date.strftime('%Y_%m_%d')}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # 添加标题
    title = Paragraph(f"Experiment Report - {date.strftime('%Y-%m-%d')}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
        # 计算总结果文件数
    total_files = sum(len(glob.glob(os.path.join(dir, "*.mrc"))) 
                     for dir in experiment['results_dirs'])
    
    # 初始化进度条
    with tqdm(total=total_files, desc=f"Processing {date.strftime('%Y-%m-%d')}", 
             unit="file", leave=False) as pbar:

    # 添加命令和结果
      for i, (command, result_dir) in enumerate(zip(experiment['commands'], experiment['results_dirs'])):
        # 添加命令标题
        story.append(Paragraph(f"Command {i+1}:", styles['Heading2']))
        
        # 添加保存目录信息
        story.append(Paragraph(f"Save Directory: {result_dir}", styles['BodyText']))
        story.append(Spacer(1, 6))
        
        # 添加命令内容
        story.append(Paragraph(command, styles['Code']))
        story.append(Spacer(1, 12))
        
        # 添加结果文件
        result_files = glob.glob(os.path.join(result_dir, "*.mrc"))


            
        # # 提取轮数和时间信息进行排序
        # def extract_sort_key(filepath):
        #     filename = os.path.basename(filepath)
        #     # 匹配rec_Y_xmin.mrc格式
        #     parts = filename.split('_')
        #     try:
        #       if parts[1]!='3d':
        #         round_num = int(parts[1])
        #       else:
        #         part2=parts[2].split('epoch')[-1]
        #         round_num = part2
        #     except:
        #         round_num=0
        #     return (round_num)
            
        #     # 按轮数和时间排序
        sorted_files = sorted(result_files, key=os.path.getmtime)
            
        story.append(Paragraph(f"Generated MRC Files ({len(sorted_files)}):", styles['Heading3']))
            


        # story.append(Paragraph(f"Generated MRC Files ({len(result_files)}):", styles['Heading3']))
        
        for mrc_file in sorted_files:
            # 显示相对路径
            #rel_path = os.path.relpath(mrc_file, start=os.path.join("results", date.strftime("%Y_%b_%d")))
            rel_path=mrc_file
            #print(rel_path)
            story.append(Paragraph(f"• {rel_path}", styles['BodyText']))
            temp_image_name=".temp"+mrc_file.replace('/','_').replace('mrc','png')
            #print(temp_image_name)
            # 生成三视图
            fig = generate_mrc_views(mrc_file)
            img_path = os.path.join(output_dir, temp_image_name)
          
            fig.savefig(img_path)
            plt.close(fig)
            
            # 添加图像
            story.append(Image(img_path, width=400, height=150))
            story.append(Spacer(1, 6))
            pbar.update(1)
            
            # 删除临时图像文件
            #os.remove(img_path)
        
        # 添加分隔线（如果不是最后一个命令）
        if i < len(experiment['commands']) - 1:
            story.append(Spacer(1, 12))
            story.append(Paragraph("_" * 80, styles['BodyText']))
            story.append(Spacer(1, 12))
    try:
      os.remove('reports/.temp*png')
    except:
      pass
    doc.build(story)


def main():
    script_dir = "./old_scripts"
    output_dir = "./reports"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析脚本
    experiments = parse_scripts(script_dir)
    
    # 为每个实验生成报告
    for date, experiment in experiments.items():
        create_pdf_report(date, experiment, output_dir)
        #print('')

if __name__ == "__main__":
    main()
