import os
import shutil

def rename_files_in_folder(folder_path, new_folder_path):
    # 폴더 내의 모든 파일 목록 가져오기
    files = os.listdir(folder_path)
    
    # 파일 이름을 정렬 (필요에 따라 정렬 기준을 변경할 수 있음)
    files.sort()
    
    # 파일 이름 변경
    name_index = 1
    for index, file_name in enumerate(files):
        if 'eeg' in file_name:
            # 파일의 확장자 추출
            file_extension = file_name[-8:]

            # 새로운 파일 이름 생성
            new_file_name = f"n{name_index:04d}{file_extension}"
            
            # 전체 경로 생성
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(new_folder_path, new_file_name)
            
            # 파일 이름 변경
            shutil.copy(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} to {new_file_path}")
            name_index += 1

# 사용 예시
folder_path = './mat_shhs1'  # 여기에 폴더 경로를 입력하세요
new_folder_path = './mat_shhs1_renamed'  # 여기에 폴더 경로를 입력하세요
rename_files_in_folder(folder_path, new_folder_path)