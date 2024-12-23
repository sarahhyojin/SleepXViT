function collection = aggregate_sleeptransformer(nchan)

    seq_len = 21;
    Nfold = 1;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    mat_path = './mat_snuh_100_prep/';
    listing = dir([mat_path, '*_eeg.mat']);
    load('./data_split_eval_snuh_100_prep.mat');

    acc_novote = [];

    for fold = 1 : Nfold
        fold
        %test_s = test_sub{fold};
        test_s = test_sub;
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            i
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            % handle the different here
            sample_size(i) = numel(label) - (seq_len - 1);
            yt{fold} = [yt{fold}; double(label)];
        end
        
        load(['./scratch_training/sleeptransformer/scratch_training_1chan_snuh_100_prep/n1/test_ret.mat']);
        
        acc_novote = [acc_novote; acc];
        
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            score_{n} = softmax(squeeze(score(:,n,:)));
        end
        score = score_;
        clear score_;

        for i = 1 : numel(test_s)
            start_pos = sum(sample_size(1:i-1)) + 1;
            end_pos = sum(sample_size(1:i-1)) + sample_size(i);
            score_i = cell(1,seq_len);
            %valid_ind = cell(1,seq_len);
            for n = 1 : seq_len
                disp(n);
                disp(i);
                disp(sample_size(i));
                disp(start_pos);
                disp(end_pos);
                disp(size(score{n}));
                score_i{n} = score{n}(start_pos:end_pos, :);
                N = size(score_i{n},1);
                %valid_ind{n} = ones(N,1);

                score_i{n} = [ones(seq_len-1,5); score{n}(start_pos:end_pos, :)];
                %valid_ind{n} = [zeros(seq_len-1,1); valid_ind{n}]; 
                score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
                %valid_ind{n} = circshift(valid_ind{n}, -(seq_len - n), 1);
            end

            smoothing = 0;
            %fused_score = score_i{1};
            %fused_score = log(score_i{1}.*repmat(valid_ind{1},1,5));
            fused_score = log(score_i{1});
            for n = 2 : seq_len
                if(smoothing == 0)
                    %fused_score = fused_score + log(score_i{n}.*repmat(valid_ind{n},1,5));
                    fused_score = fused_score + log(score_i{n});
                else
                    %fused_score = fused_score + score_i{n}.*repmat(valid_ind{n},1,5);
                    fused_score = fused_score + score_i{n};
                end
            end

            yhat = zeros(1,size(fused_score,1));
            for k = 1 : size(fused_score,1)
                [~, yhat(k)] = max(fused_score(k,:));
            end

            yh{fold} = [yh{fold}; double(yhat')];
        end
    end

    disp(acc_novote);

    yh = cell2mat(yh);
    yt = cell2mat(yt);
    % 고유값 구하기
    unique_yh = unique(yh);
    unique_yt = unique(yt);

    % 결과 출력
    disp('yh의 고유값:');
    disp(unique_yh);

    disp('yt의 고유값:');
    disp(unique_yt);

    acc = sum(yh == yt)/numel(yt)
    disp(size(yt));
    disp(size(yh));
    C = confusionmat(yt, yh);
    
    [mysensitivity, myselectivity]  = calculate_sensitivity_selectivity(yt, yh);
    
    [fscore, sensitivity, specificity] = litis_class_wise_f1(yt, yh);
    mean_fscore = mean(fscore)
    mean_sensitivity = mean(sensitivity)
    mean_specificity = mean(specificity)
    kappa = kappaindex(yh,yt,5)
    
    % count the number of samples for each class
    classes = unique(yt)
    fscore
    class_counts = histc(yt, classes)
    
    % calculate Weighted F1 Score
    weighted_f1 = sum(fscore .* class_counts) / sum(class_counts)
    weighted_f1

    str = '';
    % acc
    str = [str, '$', num2str(acc*100, '%.1f'), '$', ' & '];
    % kappa
    str = [str, '$', num2str(kappa, '%.3f'), '$', ' & '];
    % fscore
    str = [str, '$', num2str(mean_fscore*100, '%.2f'), '$', ' & '];
    % mean_sensitivity
    str = [str, '$', num2str(mean_sensitivity*100, '%.2f'), '$', ' & '];
    % mean_specificity
    str = [str, '$', num2str(mean_specificity*100, '%.2f'), '$', ' & '];
    
    % class-wise MF1
    for i = 1 : 5
        str = [str, '$', num2str(fscore(i)*100,'%.2f'), '$ & '];
    end
    str
    
    collection = [acc, mean_fscore, weighted_f1, kappa, mean_sensitivity, mean_specificity];
end
