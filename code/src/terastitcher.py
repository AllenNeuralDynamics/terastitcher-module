import platform
import os
import subprocess
from typing import List, Optional
from utils import utils

def helper_get_number_processes_align_step(config_params:dict) -> int:
    """
    
        Get the estimate number of processes to partition the dataset and calculate the align step.
        Using MPI, check if the number of slots are enough for running the number of processes.
        You can automatically set --use-hwthread-cpus to automatically estimate the number 
        of hardware threads in each core and increase the allowed number of processes. There is another
        option with -oversubscribe.
        
        Parameters:
        - config_params (dict): Parameters that will be used in the align step. i.e. {'depth': 4200, 'subvoldim': 200, 'max_num_procs': 40}
        
        Returns:
        - int: Number of processes to be used in the align step. -1 indicates that there are not optimal processes for the provided configuration.
    
    """
    partitioning_depth = math.ceil( config_params['depth'] / config_params['subvoldim'] )

    for proc in range(config_params['max_num_procs'], -1, -1):
        tiling_proc = 2 * ( proc - 1)
        
        if partitioning_depth > tiling_proc:
            return proc
            
    return -1

def helper_build_param_value_command(params:dict) -> str:
    """
    """
    parameters = ''
    for (param, value) in params.items():
        if type(value) in [str, float, int]:
            parameters += f"--{param}={value} "
            
    return parameters

def helper_additional_params_command(params:list) -> str:
    """
    """
    additional_params = ''
    for param in params:
        additional_params += f"--{param} "
        
    return additional_params

class TeraStitcher():
    
    def __init__(self, 
            input_data:str, 
            output_folder:str, 
            parallel:bool=True,
            parastitcher_path:Optional[str]=None,
            computation:Optional[str]='cpu',
        ) -> None:
        
        """
        Constructor
        
        Parameters
        ------------------------
            - 
        
        """
        
        self.__input_data = input_data
        self.__output_folder = output_folder
        self.__parallel = parallel
        self.__computation = computation
        self.__platform = platform.system()
        self.__parastitcher_path = parastitcher_path
        
        if computation not in ['cpu', 'gpu']:
            print("Setting computation to gpu")
            self.__computation = 'cpu'
        
        if not self.__check_installation():
            raise 
            print(f"Please, check your terastitcher installation in the system {self.__platform}")
        
        self.__check_parastitcher()
        
        utils.create_folder(self.__output_folder + "/xmls")
        utils.create_folder(self.__output_folder + "/metadata")
    
    def __check_installation(self, tool_name:str="terastitcher") -> bool:
        """
        
        """
        
        try:
            devnull = open(os.devnull)
            subprocess.Popen([tool_name], stdout=devnull, stderr=devnull).communicate()
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                return False
        return True
    
    def __check_parastitcher(self) -> None:
        """
        """
        
        if self.__parastitcher_path != None:
            if not os.path.exists(self.__parastitcher_path):
                raise FileNotFoundError("Parastitcher path not found.")
            else:
                # TODO Check if parastitcher works with GPU, if it does then this would not be necessary
                self.__computation = 'cpu'
        else:
            self.__parallel = False
    
    def __build_parallel_command(self, params:dict) -> str:
        """
        """
        
        if not self.__parallel:
            return ''
        
        mpi_command = 'mpiexec -n' if self.__platform == 'Windows' else 'mpirun -np'
        
        additional_params = ''
        
        if len(params['additional_params']):
            additional_params = helper_additional_params_command(params['additional_params'])
            
        cmd = f"{mpi_command} {params['number_processes']} --hostfile {params['hostfile_path']} {additional_params}"
        cmd += f"python {self.__parastitcher_path}"
        return cmd
    
    def __import_step_cmd(self, params:dict) -> str:
        """
        
        """
        # TODO Check if params comes with all necessary keys so it does not raise KeyNotFound error
        volume_input = f"--volin={self.__input_data}"
        output_folder = f"--projout={self.__output_folder}/xmls/xml_import.xml"
        
        parameters = helper_build_param_value_command(params)
        
        additional_params = ''
        if len(params['additional_params']):
            additional_params = helper_additional_params_command(params['additional_params'])
        
        cmd = f"terastitcher --import {volume_input} {output_folder} {parameters} {additional_params}"
        
        utils.save_dict_as_json(f"{self.__output_folder}/metadata/import_params.json", params, True)
        
        return cmd
    
    def __align_step_cmd(self, params:dict):
        """
        """
        
        # TODO Check the best number of processes using formula
        input_xml = f"--projin={self.__output_folder}/xmls/xml_import.xml"
        output_xml = f"--projout={self.__output_folder}/xmls/xml_displcomp_par2.xml"
        parallel_command = ''
        
        if self.__parallel:
            
            if self.__computation == 'cpu':
                # mpirun for linux and mac OS
                parallel_command = self.__build_parallel_command(params['cpu_params'])
                
            else:
                # gpu computation
                # TODO set gpu flag
                pass
        
        else:
            # Sequential execution
            parallel_command = f"terastitcher"
        
        parameters = helper_build_param_value_command(params)
        
        cmd = f"{parallel_command} --displcompute {input_xml} {output_xml} {parameters}"
        utils.save_dict_as_json(f"{self.__output_folder}/metadata/align_params.json", params, True)
        
        return cmd
    
    def __input_output_step_cmd(self, step_name:str, input_xml:str, output_xml:str) -> str:
        input_xml = f"--projin={self.__output_folder}/xmls/{input_xml}"
        output_xml = f"--projout={self.__output_folder}/xmls/{output_xml}"
        
        cmd = f"terastitcher --{step_name} {input_xml} {output_xml}"
        return cmd
        
    def execute_pipeline(self, config:dict) -> None:
        
        """
        
        """
        
        # Step 1
        print(self.__import_step_cmd(config['import']))
        
        # Step 2
        print(self.__align_step_cmd(config['align']))
        
        # Step 3
        print(self.__input_output_step_cmd('placetiles', 'xml_displthres.xml', 'xml_merging.xml'))
        
        # Step 4
        print(self.__input_output_step_cmd('displthres', 'xml_displproj.xml', 'xml_displthres.xml'))
        
        # Step 5
        print(self.__input_output_step_cmd('displproj', 'xml_displcomp_par2.xml', 'xml_displproj.xml'))
        
        # Step 6
        
    
def main():
    input_data = "C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TestData/mouse.cerebellum.300511.sub3/tomo300511_subv3"
    output_folder = "C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TestData/output_python"
    
    # TODO if we pass another path that exists instead of parastitcher's path, it builds the command
    parastitcher_path = 'C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TeraStitcher-portable-1.11.10-win64/pyscripts/Parastitcher.py'
    
    terastitcher_tool = TeraStitcher(
        input_data=input_data,
        output_folder=output_folder,
        parallel=True,
        computation='cpu',
        parastitcher_path=parastitcher_path
    )

    config = {
        "import" : {
            'ref1':'H',
            'ref2':'V',
            'ref3':'D',
            'vxl1':'1.800',
            'vxl2':'1.800',
            'vxl3':'2',
            'additional_params':['sparse_data', 'libtiff_uncompress'] # 'rescan'
        },
        "align" : {
            'cpu_params' : {
                'number_processes': 6,
                'hostfile_path': 'path/to/hostfile',
                'additional_params' : ['use-hwthread-cpus', 'allow-run-as-root']
            },
            'subvoldim': 100,
        }
    }
    
    terastitcher_tool.execute_pipeline(config)

if __name__ == "__main__":
    main()