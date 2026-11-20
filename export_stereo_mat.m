function export_stereo_mat(input, destination)
% 导出双目标定结果为 Python 可直接读取的数值 MAT。也可由主界面自动调用。
% export_stereo_mat(stereoParams, 'stereo_numeric.mat')
if ischar(input) || isstring(input)
    payload = load(input, '-mat');
    canonical = {'K1','D1','K2','D2','R','T','imageSize'};
    longNames = {'intrinsicMatrix1','distortionCoefficients1','intrinsicMatrix2', ...
        'distortionCoefficients2','rotationOfCamera2','translationOfCamera2','imageSize'};
    if all(isfield(payload, canonical)) || all(isfield(payload, longNames))
        if all(isfield(payload, canonical)), names = canonical; else, names = longNames; end
        values = struct();
        for j = 1:numel(names)
            values.(canonical{j}) = payload.(names{j});
        end
        if isfield(payload,'worldUnits'), values.worldUnits = char(payload.worldUnits); end
        write_parameters(values, destination);
        return
    end
    names = fieldnames(payload);
    candidates = {};
    for i = 1:numel(names)
        value = payload.(names{i});
        if isa(value, 'stereoParameters')
            candidates{end+1} = value; %#ok<AGROW>
        elseif isscalar(value) && ((isobject(value) && isprop(value, 'CameraParameters')) ...
                || (isstruct(value) && isfield(value, 'CameraParameters')))
            if isa(value.CameraParameters, 'stereoParameters')
                candidates{end+1} = value.CameraParameters; %#ok<AGROW>
            end
        end
    end
    if numel(candidates) ~= 1
        error('请选择包含一组已完成双目标定结果的 MAT 文件。也可以将指定 stereoParams 单独导出。');
    end
    params = candidates{1};
else
    params = input;
end
[K1,D1,K2,D2,R,T] = stereoParametersToOpenCV(params);
imageSize = params.CameraParameters1.ImageSize; % [height width] of one eye
assert(isequal(imageSize,params.CameraParameters2.ImageSize), '左右图像尺寸必须一致。');
values = struct('K1',K1,'D1',D1,'K2',K2,'D2',D2,'R',R,'T',T, ...
    'imageSize',imageSize,'worldUnits',char(params.CameraParameters1.WorldUnits));
write_parameters(values, destination);
end

function write_parameters(values, destination)
[~,~,ext] = fileparts(destination);
if strcmpi(ext,'.json')
    fid = fopen(destination, 'w', 'n', 'UTF-8');
    assert(fid ~= -1, '无法写入转换结果。');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', jsonencode(values));
else
    save(destination, '-struct', 'values', '-v7');
end
end
