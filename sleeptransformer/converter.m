% 폴더 경로 입력받기
input_folder = '/data2/prep_sg_snuh_mat/';
output_folder = '/data2/prep_sg_snuh_mat_converted/';
n = 1; % n의 초기값 설정

% 입력 폴더의 모든 .mat 파일 목록 가져오기
mat_files = dir(fullfile(input_folder, '*.mat'));

% 각 파일에 대해 반복
for i = 1:length(mat_files)
    % 파일 읽기
    input_file = fullfile(input_folder, mat_files(i).name);
    data = load(input_file);

    % 필요한 데이터 추출
    X1 = data.X1; % X1 변수
    X2 = data.X2; % X2 변수
    label = data.label; % label 변수
    y = data.y; % y 변수

    % 저장할 파일 이름 설정
    output_file = fullfile(output_folder, mat_files(i).name);
    
    % .mat 파일로 저장
    save(output_file, 'X1', 'X2', 'label', 'y', '-v7.3');
    
    % n 값을 증가시켜 파일 이름을 다르게 설정
    n = n + 1;
end
