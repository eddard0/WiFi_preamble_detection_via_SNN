clear; clc; close all;

%% User setting
csvFile = 'wifi_preamble.csv';

%% Read CSV
data = readmatrix(csvFile);
if isempty(data)
    error('CSV 파일이 비어 있습니다.');
end

labels = data(:, end);
idx = find(labels == 1, 1, 'first');
if isempty(idx)
    error('label=1 인 행이 없습니다.');
end

row = data(idx, :);
iq = row(1:end-1);

if mod(numel(iq), 2) ~= 0
    error('CSV 형식이 [I_1,Q_1,I_2,Q_2,...,label] 구조가 아닙니다.');
end

%% Reconstruct complex signal
I = iq(1:2:end);
Q = iq(2:2:end);
x = complex(I, Q);

%% Magnitude
mag = abs(x);
magN = mag / (max(mag) + eps);

n = 0:numel(x)-1;

%% Plot
figure;
plot(n, magN, 'b', 'LineWidth', 1.2); hold on;
xline(160, '--r', 'LineWidth', 1.0);
xline(320, '--r', 'LineWidth', 1.0);
grid on;
xlabel('Sample Index');
ylabel('Normalized Magnitude');
title(sprintf('Magnitude of one label=1 sample (row %d)', idx));

text(80,  0.92, 'STF', 'HorizontalAlignment', 'center');
text(240, 0.92, 'LTF', 'HorizontalAlignment', 'center');
text(360, 0.92, 'SIG', 'HorizontalAlignment', 'center');