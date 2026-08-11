import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QLineEdit, QProgressBar,
                             QTextEdit, QGroupBox, QMessageBox, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# 读取环境配置
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# 工具函数：执行外部脚本（区分AI环境 / ArcGIS环境）
def run_external_script(python_exe, script_path, args_list, log_signal):
    cmd = [python_exe, script_path] + args_list#执行器，脚本文件追加所有参数（--before等）以列表传给popen
    proc = subprocess.Popen(#捕获正常和报错给日志，输出内容转字符串，utf8中文不乱码
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="gbk"#windows中文系统用gbk
    )#创建子进程单独运行AI/GIS脚本，二者完全隔离不会出现库版本冲突
    while True:#循环逐行读取实时刷新日志
        line = proc.stdout.readline()#实时打印脚本输出内容
        if not line and proc.poll() is not None:#检查是否已经没有新输出且进程已退出
            break
        if line:
            log_signal.emit(line.strip())#将读取的单行日志strip去掉换行空格通过qt信号发给日志框
    err = proc.stderr.read()#循环结束一次性读取全部报错
    if err:
        log_signal.emit(f"【错误信息】{err}")
    return proc.returncode#将子进程执行结果返回调用者workthread线程（0为成功）

# 后台线程：防止界面卡死，跑AI推理与GIS分析等耗时计算
class WorkThread(QThread):
    #均为信号量用于跨线程通信
    log_msg = pyqtSignal(str)#将AI/GIS脚本打印的日志发送给界面日志框对应上面res函数最后一个参数
    progress = pyqtSignal(int)#传输数字10-100实时更新界面进度条
    finish_flag = pyqtSignal(bool)#主线程收到后判断成功/失败，恢复按钮可用状态并弹窗

    def __init__(self, task_type, params):
        super().__init__()#继承父类QThread构造逻辑
        self.task_type = task_type#选择要执行的操作：即各个功能脚本
        self.params = params#存储所有文件路径的字典

    def run(self):#重写 run () 函数，调用thread.start时会自动执行run内部代码，所有耗时逻辑在此
        try:#用try-except捕获未知异常给logmsg，不会闪退
            if self.task_type == "ai_infer":
                self.log_msg.emit("===== 启动Siam-UNet损毁识别模型 =====")
                self.progress.emit(10)#进度条信号到10%表示AI任务开始
                # 基础参数固定
                ai_args = [
                    "--before", self.params["before_tif"],
                    "--after", self.params["after_tif"],
                    "--out", self.params["damage_tif"]
                ]
                # 如果勾选TTA，追加参数 --use_tta
                if self.params["use_tta"]:
                    ai_args.append("--use_tta")

                code = run_external_script(
                    CONFIG["ai_env_python"],
                    "ai_infer.py",
                    ai_args,  # 传入动态参数列表
                    self.log_msg
                )
                self.progress.emit(30)
                if code != 0:
                    self.log_msg.emit("AI模型推理失败！")
                    self.finish_flag.emit(False)
                    return

            elif self.task_type == "building_shp":
                self.log_msg.emit("===== 启动遥感建筑自动识别模型 =====")
                self.progress.emit(40)
                # 调用文件夹里的python执行推理脚本
                ret_code = run_external_script(
                    CONFIG["ai_env_python"],
                    "building_extract.py",
                    ["--before", self.params["before_tif"]],
                    self.log_msg
                )
                self.progress.emit(60)
                if ret_code != 0:
                    self.log_msg.emit("建筑提取失败，请检查影像路径或模型权重")
                    self.finish_flag.emit(False)
                    return

            elif self.task_type == "export_map":
                self.log_msg.emit("===== 自动生成灾情分级专题图 =====")
                self.run_single_task("ai_infer", self.params)
                self.run_single_task("building_shp", self.params)
                self.progress.emit(70)
                code = run_external_script(
                    CONFIG["ai_env_python"],
                    "map_export.py",
                    [
                        "--base",self.params["before_tif"],
                        "--building",self.params["building_shp"],
                        "--damage", self.params["damage_tif"],
                        "--outtif", os.path.join(CONFIG["output_folder"], "damage_overlay.tif"),
                        "--outpng", os.path.join(CONFIG["output_folder"], "damage_preview.png"),
                        "--outstats",os.path.join(CONFIG["output_folder"], "stats.json"),
                        "--outlegend",os.path.join(CONFIG["output_folder"], "legend.txt"),
                    ],
                    self.log_msg
                )
                self.progress.emit(80)
                if code != 0:
                    self.log_msg.emit("数据导出失败！")
                    self.finish_flag.emit(False)
                    return
                
            elif self.task_type == "pdf":
                self.log_msg.emit("===== 自动生成灾情分级专题图 =====")
                self.run_single_task("map_export", self.params)
                self.progress.emit(90)
                code = run_external_script(
                    CONFIG["ai_env_python"],
                    "generate_report_pdf.py",
                    [
                        "--base",self.params["before_tif"],
                        "--building",self.params["building_shp"],
                        "--damage", self.params["damage_tif"],
                        "--outpdf", os.path.join(CONFIG["output_folder"], "damage_PDF.pdf")
                    ],
                    self.log_msg
                )
                self.progress.emit(100)
                if code != 0:
                    self.log_msg.emit("专题图导出失败！")
                    self.finish_flag.emit(False)
                    return
                """   
            elif self.task_type == "batch":
                self.log_msg.emit("===== 文件夹批量分析 =====")
                self.progress.emit(50)
                code = run_external_script(
                    CONFIG["ai_env_python"],
                    "batch_processor.py",
                    [
                        "--gdb", os.path.join(CONFIG["output_folder"], CONFIG["gdb_name"]),
                        "--road", self.params["road_shp"],
                        "--shelter", self.params["shelter_shp"]
                    ],
                    self.log_msg
                )
                self.progress.emit(100)
                if code != 0:
                    self.log_msg.emit("路网分析失败！")
                    self.finish_flag.emit(False)
                    return
                
            elif self.task_type == "all_auto":
                # 一键全流程串联执行
                self.run_single_task("ai_infer", self.params)
                self.run_single_task("gis_stat", self.params)
                self.run_single_task("export_map", self.params)
                #self.run_single_task("road_analysis", self.params)
                self.progress.emit(100)
                """
            self.log_msg.emit("==================== 全部任务执行完成 ====================")
            self.finish_flag.emit(True)
        except Exception as e:
            self.log_msg.emit(f"程序异常：{str(e)}")
            self.finish_flag.emit(False)

    def run_single_task(self, task, params):#一键全流程时避免重复写前四段代码
        sub_thread = WorkThread(task, params)
        sub_thread.log_msg.connect(lambda x: self.log_msg.emit(x))#connect将信号绑定后面的函数，
        sub_thread.progress.connect(lambda x: self.progress.emit(x))#信号发射时自动执行后者，x是信号携带的参数
        sub_thread.run()

# 主窗口界面
class MainWindow(QMainWindow):#QMainWindow 结构分菜单栏、状态栏、中心区域
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于Siam-UNet的灾后建筑损毁GIS评估系统")
        self.resize(900, 700)#窗口高宽
        self.init_ui()
        self.work_thread = None#用于保存全局的后台线程，方便信号绑定、状态管理

    def init_ui(self):
        central_widget = QWidget()#创建空白面板
        self.setCentralWidget(central_widget)#把上面创建的空白面板挂载到窗口中间，所有输入框、按钮、图表都放在这块面板上
        main_layout = QVBoxLayout(central_widget)# 垂直布局，依次 addWidget 的控件，会自动纵向堆叠

        # 1. 数据输入区域
        group_data = QGroupBox("1. 数据源输入")#所有文件选择框归一组
        layout_data = QVBoxLayout(group_data)#分组内部再垂直摆放控件

        # 灾前影像
        h1 = QHBoxLayout()#水平布局，文字框+输入框+按钮
        self.edit_before = QLineEdit()#文本框，保存选中文件路径，只能输入一行文本的输入控件
        btn_before = QPushButton("选择灾前遥感TIF")#选择按钮
        btn_before.clicked.connect(lambda: self.select_file(self.edit_before,
                                                            "TIF(*.tif)"))
        #按钮点击触发文件选择函数，把要回填的输入框、文件过滤格式传进去
        h1.addWidget(QLabel("灾前影像："))#把标签、输入框、按钮加入水平布局
        h1.addWidget(self.edit_before)
        h1.addWidget(btn_before)
        layout_data.addLayout(h1)

        # 灾后影像
        h2 = QHBoxLayout()
        self.edit_after = QLineEdit()
        btn_after = QPushButton("选择灾后遥感TIF")
        btn_after.clicked.connect(lambda: self.select_file(self.edit_after, "TIF(*.tif)"))
        h2.addWidget(QLabel("灾后影像："))
        h2.addWidget(self.edit_after)
        h2.addWidget(btn_after)
        layout_data.addLayout(h2)
        main_layout.addWidget(group_data)

        """# 建筑矢量,支持外部下载
        h3 = QHBoxLayout()
        self.edit_building = QLineEdit()
        btn_building = QPushButton("建筑轮廓SHP")
        btn_building.clicked.connect(lambda: self.select_file(self.edit_building, "SHP(*.shp)"))
        h3.addWidget(QLabel("建筑面矢量："))
        h3.addWidget(self.edit_building)
        h3.addWidget(btn_building)
        layout_data.addLayout(h3)
        main_layout.addWidget(group_data)

        # 行政区
        h4 = QHBoxLayout()
        self.edit_district = QLineEdit()
        btn_district = QPushButton("行政区边界SHP")
        btn_district.clicked.connect(lambda: self.select_file(self.edit_district, "SHP(*.shp)"))
        h4.addWidget(QLabel("行政区矢量："))
        h4.addWidget(self.edit_district)
        h4.addWidget(btn_district)
        layout_data.addLayout(h4)
        
        # 路网
        h5 = QHBoxLayout()
        self.edit_road = QLineEdit()
        btn_road = QPushButton("道路路网SHP")
        btn_road.clicked.connect(lambda: self.select_file(self.edit_road, "SHP(*.shp)"))
        h5.addWidget(QLabel("路网矢量："))
        h5.addWidget(self.edit_road)
        h5.addWidget(btn_road)
        layout_data.addLayout(h5)
        main_layout.addWidget(group_data)

        # ========== 新增：救助站点矢量 ==========
        h7 = QHBoxLayout()
        self.edit_shelter = QLineEdit()
        btn_shelter = QPushButton("救助站点SHP")
        # 绑定文件选择函数，只筛选shp文件
        btn_shelter.clicked.connect(lambda: self.select_file(self.edit_shelter, "SHP(*.shp)"))
        h7.addWidget(QLabel("救助站点："))
        h7.addWidget(self.edit_shelter)
        h7.addWidget(btn_shelter)
        layout_data.addLayout(h7)
        """
        # TTA选择
        h3 = QHBoxLayout()
        self.check_tta = QCheckBox("启用TTA高精度推理(速度更慢)")#复选框控件
        self.check_tta.setChecked(True)  # 默认勾选，提高精度
        h3.addWidget(self.check_tta)#start_task 里用 .isChecked() 读取布尔值，传给线程参数
        layout_data.addLayout(h3)#把控件放进布局里，自动排版

        # 2. 功能按钮区
        group_func = QGroupBox("2. 功能工具箱")
        layout_func = QHBoxLayout(group_func)
        self.btn_ai = QPushButton("①AI损毁识别推理")
        self.btn_shp = QPushButton("②建筑shp轮廓提取")
        self.btn_map = QPushButton("③生成灾情数据")
        self.btn_pdf = QPushButton("生成专题图")
        #self.btn_road = QPushButton("④路网可达性分析")
        #self.btn_all = QPushButton("★一键全自动完整流程")
        # 绑定槽函数，点击全部绑定同一个通用函数start_task，代码高度复用
        self.btn_ai.clicked.connect(lambda: self.start_task("ai_infer"))
        self.btn_shp.clicked.connect(lambda: self.start_task("building_shp"))
        self.btn_map.clicked.connect(lambda: self.start_task("export_map"))
        self.btn_pdf.clicked.connect(lambda: self.start_task("pdf"))
        #self.btn_road.clicked.connect(lambda: self.start_task("road_analysis"))
        #self.btn_all.clicked.connect(lambda: self.start_task("all_auto"))
        layout_func.addWidget(self.btn_ai)
        layout_func.addWidget(self.btn_shp)
        layout_func.addWidget(self.btn_map)
        layout_func.addWidget(self.btn_pdf)
        #layout_func.addWidget(self.btn_road)
        #layout_func.addWidget(self.btn_all)
        main_layout.addWidget(group_func)

        # 3. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main_layout.addWidget(QLabel("执行进度："))
        main_layout.addWidget(self.progress_bar)

        # 4. 运行日志窗口
        group_log = QGroupBox("3. 运行日志")
        log_layout = QVBoxLayout(group_log)
        self.log_text = QTextEdit()#多行文本框
        self.log_text.setReadOnly(True)#只读文本框
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(group_log)

    # 文件选择弹窗
    def select_file(self, line_edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "./test", filter_str)
        #弹出系统文件选择窗口，默认打开./test_data样例文件夹，filter_str 文件过滤TIF(*.tif) 只显示 tif
        if path:
            line_edit.setText(path)#选中文件后回填路径的输入框

    # 启动后台任务，参数统一封装
    def start_task(self, task_type):
        #读取界面所有输入
        before_tif = self.edit_before.text().strip()
        after_tif = self.edit_after.text().strip()
        #building_shp = self.edit_building.text().strip()
        #district_shp = self.edit_district.text().strip()
        #road_shp = self.edit_road.text().strip()

        # 基础校验
        if not before_tif or not after_tif:
            QMessageBox.warning(self, "提示", "请先选择灾前、灾后遥感影像！")
            return#不创建线程，提前拦截错误
        damage_tif = os.path.join(CONFIG["output_folder"], "damage_result.tif")#推理得到的灰度图
        building_shp = os.path.join(CONFIG["output_folder"], "building.shp")
        params = {
            "before_tif": before_tif,
            "after_tif": after_tif,
            "damage_tif": damage_tif,
            "building_shp": building_shp,
            #"district_shp": district_shp,
            #"road_shp": road_shp,
            #"shelter_shp": self.edit_shelter.text().strip(),
            # 新增：读取复选框勾选状态 True/False
            "use_tta": self.check_tta.isChecked()
        }#文件路径与勾选状态全部打包给workthread
        # 清空日志、重置进度
        self.log_text.clear()
        self.progress_bar.setValue(0)
        # 禁用按钮防止重复点击
        self.set_btn_status(False)
        # 创建并启动线程
        self.work_thread = WorkThread(task_type, params)
        self.work_thread.log_msg.connect(self.append_log)#绑定append_log，往文本框追加文字
        self.work_thread.progress.connect(self.progress_bar.setValue)#绑定进度条 setValue，自动更新百分比
        self.work_thread.finish_flag.connect(self.task_finished)#绑定task_finished，任务结束弹窗、解锁按钮
        self.work_thread.start()#自动执行run

    def append_log(self, msg):
        self.log_text.append(msg)

    def task_finished(self, success):
        self.set_btn_status(True)#无论成功失败，先解锁所有按钮
        if success:
            QMessageBox.information(self, "完成", "当前任务全部执行完毕！成果保存在output_data文件夹")
        else:
            QMessageBox.critical(self, "失败", "任务执行出错，请查看日志排查问题！")

    def set_btn_status(self, enable):
        self.btn_ai.setEnabled(enable)
        self.btn_shp.setEnabled(enable)
        self.btn_map.setEnabled(enable)
        self.btn_pdf.setEnabled(enable)
        #self.btn_road.setEnabled(enable)
        #self.btn_all.setEnabled(enable)

if __name__ == "__main__":#直接双击运行时下方会运行，import main则不会
    app = QApplication(sys.argv)#Qt 图形程序的全局管理核心对象，传入命令行参数["main.py", "--test"]
    window = MainWindow()
    window.show()
    sys.exit(app.exec())#app.exec()执行后程序不会立刻结束，进入无限等待状态，
                        #只有关闭所有窗口后才会结束运行，同时返回一个数字0代表正常关闭
