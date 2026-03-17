function result = capture_wifi_preamble_n210(varargin)
% capture_wifi_preamble_binary_separated_n210
%
% Two-phase capture for a binary Wi-Fi dataset with USRP N210.
%
% Phase 1:
%   Capture aligned legacy Wi-Fi preamble clips
%   label = 1
%
% Phase 2:
%   Capture noise-only clips in a separate run
%   label = 0
%
% Final output:
%   Merge phase-1 and phase-2 clips
%   Shuffle sample order
%   Save one CSV/MAT dataset
%
% CSV row format:
%   [I_001, Q_001, I_002, Q_002, ..., I_400, Q_400, label]
%
% Example:
% result = capture_wifi_preamble_binary_separated_n210( ...
%     'RadioIP', '192.168.10.2', ...
%     'CenterFrequency', 2.412e9, ...
%     'SampleRate', 20e6, ...
%     'WiFiChannelBandwidth', 'CBW20', ...
%     'Gain', 18, ...
%     'NumPositive', 30000, ...
%     'NumNegative', 30000, ...
%     'OutputFile', 'wifi_preamble_binary');

%% Parameters
p = inputParser;

addParameter(p, 'RadioIP', '192.168.10.2', @(x)ischar(x) || isstring(x));
addParameter(p, 'CenterFrequency', 2.412e9, @(x)isnumeric(x) && isscalar(x));
addParameter(p, 'Gain', 18, @(x)isnumeric(x) && isscalar(x));
addParameter(p, 'MasterClockRate', 100e6, @(x)isnumeric(x) && isscalar(x));
addParameter(p, 'SampleRate', 20e6, @(x)isnumeric(x) && isscalar(x));
addParameter(p, 'WiFiChannelBandwidth', 'CBW20', @(x)ischar(x) || isstring(x));

addParameter(p, 'SamplesPerFrame', 6000, @(x)isnumeric(x) && isscalar(x) && x >= 800);
addParameter(p, 'OverlapLength', 1200, @(x)isnumeric(x) && isscalar(x) && x >= 400);
addParameter(p, 'ClipLength', 400, @(x)isnumeric(x) && isscalar(x) && x == 400);

addParameter(p, 'NumPositive', 500, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'NumNegative', 500, @(x)isnumeric(x) && isscalar(x) && x >= 0);

addParameter(p, 'PacketDetectThreshold', 0.8, @(x)isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
addParameter(p, 'SymbolTimingThreshold', 0.5, @(x)isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
addParameter(p, 'FineSearchBackoff', 96, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'FineSearchForward', 900, @(x)isnumeric(x) && isscalar(x) && x >= 400);
addParameter(p, 'PositiveMinSeparation', 200, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'PositiveSTFMetricMin', 0.55, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'PositiveLocalDetectTolerance', 64, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'MinPositivePower', 1e-8, @(x)isnumeric(x) && isscalar(x) && x >= 0);

addParameter(p, 'MaxNegativePower', inf, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'MaxNegativeSTFMetric', 0.20, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'NegativeMinSeparation', 200, @(x)isnumeric(x) && isscalar(x) && x >= 0);

addParameter(p, 'UseBurstMode', true, @(x)islogical(x) || isnumeric(x));
addParameter(p, 'NumFramesInBurst', 1, @(x)isnumeric(x) && isscalar(x) && x >= 1);

% Per-attempt frame budgets
addParameter(p, 'FramesPerAttemptPositive', 30000, @(x)isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'FramesPerAttemptNegative', 30000, @(x)isnumeric(x) && isscalar(x) && x >= 1);

addParameter(p, 'ResetAfterConsecutiveFailures', 3, @(x)isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'MaxReceiverResets', 30, @(x)isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'PauseAfterReset', 0.2, @(x)isnumeric(x) && isscalar(x) && x >= 0);

addParameter(p, 'VerboseEvery', 10, @(x)isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'HeartbeatEveryFrames', 10, @(x)isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'RandomSeed', 1234, @(x)isnumeric(x) && isscalar(x));

% Phase switch
addParameter(p, 'AutoStartPhase2', true, @(x)islogical(x) || isnumeric(x));
addParameter(p, 'PhaseSwitchDelaySec', 0, @(x)isnumeric(x) && isscalar(x) && x >= 0);

addParameter(p, 'SaveFormat', 'csv', @(x)ischar(x) || isstring(x));
addParameter(p, 'CSVChunkSize', 1000, @(x)isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'OutputFile', 'wifi_preamble_inf', @(x)ischar(x) || isstring(x));

parse(p, varargin{:});
cfg = p.Results;

cfg.RadioIP = char(cfg.RadioIP);
cfg.WiFiChannelBandwidth = upper(char(cfg.WiFiChannelBandwidth));
cfg.SaveFormat = lower(char(cfg.SaveFormat));
cfg.OutputFile = char(cfg.OutputFile);

if ~ismember(cfg.WiFiChannelBandwidth, {'CBW5','CBW10','CBW20'})
    error('WiFiChannelBandwidth must be one of CBW5, CBW10, or CBW20.');
end
if ~ismember(cfg.SaveFormat, {'mat','csv','both'})
    error('SaveFormat must be one of mat, csv, or both.');
end

ratio = cfg.MasterClockRate / cfg.SampleRate;
if abs(round(ratio) - ratio) > 1e-12
    error('MasterClockRate / SampleRate must be an integer.');
end
cfg.DecimationFactor = round(ratio);

expectedFs = localExpectedSampleRate(cfg.WiFiChannelBandwidth);
if abs(cfg.SampleRate - expectedFs) > 1
    warning('SampleRate %.3f MHz does not match %s nominal rate %.3f MHz.', ...
        cfg.SampleRate/1e6, cfg.WiFiChannelBandwidth, expectedFs/1e6);
end

if cfg.OverlapLength >= cfg.SamplesPerFrame
    error('OverlapLength must be smaller than SamplesPerFrame.');
end

rng(cfg.RandomSeed);

fprintf('=== Two-phase Wi-Fi binary dataset capture ===\n');
fprintf('Radio IP                 : %s\n', cfg.RadioIP);
fprintf('Center frequency         : %.6f GHz\n', cfg.CenterFrequency/1e9);
fprintf('Wi-Fi bandwidth          : %s\n', cfg.WiFiChannelBandwidth);
fprintf('Sample rate              : %.3f MHz\n', cfg.SampleRate/1e6);
fprintf('RX gain                  : %.1f dB\n', cfg.Gain);
fprintf('Clip length              : %d\n', cfg.ClipLength);
fprintf('Target positives         : %d\n', cfg.NumPositive);
fprintf('Target negatives         : %d\n', cfg.NumNegative);

%% Connection checks
fprintf('Checking radio connections...\n');
try
    radios = findsdru;
    if ~isempty(radios)
        disp(radios);
    end
catch ME
    warning('findsdru failed: %s', ME.message);
end

try
    [~, status] = probesdru(cfg.RadioIP);
    if status ~= 0
        warning('probesdru returned status = %d', status);
    end
catch ME
    warning('probesdru failed: %s', ME.message);
end

%% Phase 1: positive capture
fprintf('\n=== Phase 1: positive capture (Wi-Fi ON) ===\n');
posData = localCapturePositivePhase(cfg);

%% Phase switch
fprintf('\n=== Switch to noise-only environment ===\n');
fprintf('Turn OFF the Wi-Fi transmitter or move to a packet-free/noise-only condition.\n');

if cfg.PhaseSwitchDelaySec > 0
    fprintf('Waiting %.1f seconds before Phase 2...\n', cfg.PhaseSwitchDelaySec);
    pause(cfg.PhaseSwitchDelaySec);
end

if logical(cfg.AutoStartPhase2)
    fprintf('Phase 2 will start automatically.\n');
else
    input('When ready, press Enter to start Phase 2...', 's');
end

%% Phase 2: negative capture
fprintf('\n=== Phase 2: negative capture (noise only) ===\n');
negData = localCaptureNegativePhase(cfg);

%% Merge + shuffle
x = [posData.x; negData.x];
y = [ones(size(posData.x,1),1,'uint8'); zeros(size(negData.x,1),1,'uint8')];

info = struct();
info.phase = [ones(size(posData.x,1),1); 2*ones(size(negData.x,1),1)];
info.frameIndex = [posData.info.frameIndex; negData.info.frameIndex];
info.startIdx   = [posData.info.startIdx;   negData.info.startIdx];
info.clipPower  = [posData.info.clipPower;  negData.info.clipPower];
info.stfMetric  = [posData.info.stfMetric;  negData.info.stfMetric];
info.label      = double(y);

if ~isempty(y)
    perm = randperm(numel(y));
    x = x(perm,:);
    y = y(perm);
    info.phase      = info.phase(perm);
    info.frameIndex = info.frameIndex(perm);
    info.startIdx   = info.startIdx(perm);
    info.clipPower  = info.clipPower(perm);
    info.stfMetric  = info.stfMetric(perm);
    info.label      = info.label(perm);
end

L = cfg.ClipLength;
x_ri = zeros(size(x,1), L, 2, 'single');
x_ri(:,:,1) = real(x);
x_ri(:,:,2) = imag(x);

[outputFolder, outputStem] = localResolveOutputStem(cfg.OutputFile);
matFile = fullfile(outputFolder, [outputStem '.mat']);
csvFile = fullfile(outputFolder, [outputStem '.csv']);

if ismember(cfg.SaveFormat, {'mat','both'})
    save(matFile, 'x', 'x_ri', 'y', 'info', '-v7.3');
end
if ismember(cfg.SaveFormat, {'csv','both'})
    localWriteCsvDataset(csvFile, x, y, cfg.CSVChunkSize);
end

fprintf('\nDone.\n');
if ismember(cfg.SaveFormat, {'mat','both'})
    fprintf('Saved MAT file          : %s\n', matFile);
end
if ismember(cfg.SaveFormat, {'csv','both'})
    fprintf('Saved CSV file          : %s\n', csvFile);
    fprintf('CSV layout              : [I_001, Q_001, ..., I_400, Q_400, label]\n');
end
fprintf('Saved positives         : %d / %d\n', size(posData.x,1), cfg.NumPositive);
fprintf('Saved negatives         : %d / %d\n', size(negData.x,1), cfg.NumNegative);

result = struct();
result.matFile = '';
result.csvFile = '';
if ismember(cfg.SaveFormat, {'mat','both'})
    result.matFile = matFile;
end
if ismember(cfg.SaveFormat, {'csv','both'})
    result.csvFile = csvFile;
end
result.numPositive = size(posData.x,1);
result.numNegative = size(negData.x,1);

end

%% ===================== Positive phase =====================
function out = localCapturePositivePhase(cfg)

Lp = cfg.NumPositive;
L  = cfg.ClipLength;

xPos = complex(zeros(Lp, L, 'single'));

info.frameIndex = zeros(Lp,1,'uint32');
info.startIdx   = zeros(Lp,1,'uint64');
info.clipPower  = zeros(Lp,1,'single');
info.stfMetric  = zeros(Lp,1,'single');

numPos = 0;
numFramesTotal = 0;
attemptIdx = 0;

totalOverrun = 0;
receiverResets = 0;
lastProgressFrame = 0;

totalSamplesRead = uint64(0);
lastAcceptedPositiveAbsStart = int64(-1e12);
tail = complex(zeros(0,1,'single'));

fprintf('Phase 1 capture loop started.\n');

while numPos < Lp
    attemptIdx = attemptIdx + 1;
    fprintf('\n[Phase1] Attempt %d started...\n', attemptIdx);

    rx = localCreateReceiver(cfg);
    cleanupObj = onCleanup(@() localReleaseReceiver(rx)); %#ok<NASGU>

    consecutiveFailures = 0;
    framesThisAttempt = 0;

    while numPos < Lp && framesThisAttempt < cfg.FramesPerAttemptPositive
        try
            [rxSig, len, overrun] = rx();
            consecutiveFailures = 0;
        catch ME
            consecutiveFailures = consecutiveFailures + 1;
            warning('Positive-phase rx failure (%d/%d): %s', ...
                consecutiveFailures, cfg.ResetAfterConsecutiveFailures, ME.message);

            if consecutiveFailures >= cfg.ResetAfterConsecutiveFailures
                receiverResets = receiverResets + 1;
                warning('[Phase1] Receiver reset %d', receiverResets);
                localReleaseReceiver(rx);
                pause(cfg.PauseAfterReset);
                rx = localCreateReceiver(cfg);
                consecutiveFailures = 0;
                tail = complex(zeros(0,1,'single'));
            end
            drawnow limitrate;
            continue;
        end

        framesThisAttempt = framesThisAttempt + 1;
        numFramesTotal = numFramesTotal + 1;
        totalOverrun = totalOverrun + overrun;

        if len <= 0 || isempty(rxSig)
            drawnow limitrate;
            continue;
        end

        rxSig = single(rxSig(:));
        rxSig = rxSig(1:len);

        block = [tail; rxSig];
        block = block - mean(block);

        nTail = numel(tail);
        nBlock = numel(block);
        blockAbsStart = int64(totalSamplesRead) - int64(nTail) + 1;

        if nBlock < L
            tail = block;
            totalSamplesRead = totalSamplesRead + uint64(len);
            continue;
        end

        coarseStarts = localPacketDetectAll(block, cfg.WiFiChannelBandwidth, cfg.PacketDetectThreshold, 160);

        for k = 1:numel(coarseStarts)
            if numPos >= Lp
                break;
            end

            coarseIdx = coarseStarts(k);
            searchStart = max(1, coarseIdx - cfg.FineSearchBackoff);
            searchEnd   = min(nBlock, coarseIdx + cfg.FineSearchForward);
            searchSig   = block(searchStart:searchEnd);

            fineOff = localSymbolTiming(searchSig, cfg.WiFiChannelBandwidth, cfg.SymbolTimingThreshold);
            if isempty(fineOff) || fineOff < 0
                continue;
            end

            st = searchStart + fineOff;
            ed = st + L - 1;
            if st < 1 || ed > nBlock
                continue;
            end

            absStart = blockAbsStart + int64(st) - 1;
            if absStart <= lastAcceptedPositiveAbsStart + int64(cfg.PositiveMinSeparation)
                continue;
            end

            clip = block(st:ed);
            clipPow = mean(abs(clip).^2);
            if clipPow < cfg.MinPositivePower
                continue;
            end

            stfMetric = localSTFMetric(clip);
            if stfMetric < cfg.PositiveSTFMetricMin
                continue;
            end

            localDetect0 = localPacketDetectFirst(clip, cfg.WiFiChannelBandwidth, cfg.PacketDetectThreshold);
            if isempty(localDetect0) || localDetect0 > cfg.PositiveLocalDetectTolerance
                continue;
            end

            numPos = numPos + 1;
            lastAcceptedPositiveAbsStart = absStart;

            xPos(numPos,:) = reshape(clip, 1, []);
            info.frameIndex(numPos) = uint32(numFramesTotal);
            info.startIdx(numPos)   = uint64(absStart);
            info.clipPower(numPos)  = single(clipPow);
            info.stfMetric(numPos)  = single(stfMetric);
            lastProgressFrame = numFramesTotal;

            if mod(numFramesTotal, cfg.HeartbeatEveryFrames) == 100
                fprintf('Positive %d / %d | frame=%d | absStart=%d | power=%.4e | STF=%.3f\n', ...
                    numPos, Lp, numFramesTotal, absStart, clipPow, stfMetric);
            end
        end

        keep = min(cfg.OverlapLength, nBlock);
        tail = block(end-keep+1:end);
        totalSamplesRead = totalSamplesRead + uint64(len);

        if mod(numFramesTotal, cfg.HeartbeatEveryFrames) == 0
            fprintf('Phase1 heartbeat | attempt=%d | frame=%d | pos=%d/%d | coarseDet=%d | overruns=%d | resets=%d | lastProgress=%d\n', ...
                attemptIdx, numFramesTotal, numPos, Lp, numel(coarseStarts), totalOverrun, receiverResets, lastProgressFrame);
        end

        drawnow limitrate;
    end

    localReleaseReceiver(rx);

    if numPos < Lp
        fprintf('[Phase1] Attempt %d ended without enough positives. Restarting capture...\n', attemptIdx);
        pause(cfg.PauseAfterReset);
    end
end

xPos = xPos(1:numPos,:);
info.frameIndex = double(info.frameIndex(1:numPos));
info.startIdx   = double(info.startIdx(1:numPos));
info.clipPower  = double(info.clipPower(1:numPos));
info.stfMetric  = double(info.stfMetric(1:numPos));

out = struct();
out.x = xPos;
out.info = info;

end

%% ===================== Negative phase =====================
function out = localCaptureNegativePhase(cfg)

Ln = cfg.NumNegative;
L  = cfg.ClipLength;

xNeg = complex(zeros(Ln, L, 'single'));

info.frameIndex = zeros(Ln,1,'uint32');
info.startIdx   = zeros(Ln,1,'uint64');
info.clipPower  = zeros(Ln,1,'single');
info.stfMetric  = zeros(Ln,1,'single');

numNeg = 0;
numFramesTotal = 0;
attemptIdx = 0;

totalOverrun = 0;
receiverResets = 0;
lastProgressFrame = 0;

totalSamplesRead = uint64(0);
lastAcceptedNegativeAbsStart = int64(-1e12);
tail = complex(zeros(0,1,'single'));

fprintf('Phase 2 capture loop started.\n');

while numNeg < Ln
    attemptIdx = attemptIdx + 1;
    fprintf('\n[Phase2] Attempt %d started...\n', attemptIdx);

    rx = localCreateReceiver(cfg);
    cleanupObj = onCleanup(@() localReleaseReceiver(rx)); %#ok<NASGU>

    consecutiveFailures = 0;
    framesThisAttempt = 0;

    while numNeg < Ln && framesThisAttempt < cfg.FramesPerAttemptNegative
        try
            [rxSig, len, overrun] = rx();
            consecutiveFailures = 0;
        catch ME
            consecutiveFailures = consecutiveFailures + 1;
            warning('Negative-phase rx failure (%d/%d): %s', ...
                consecutiveFailures, cfg.ResetAfterConsecutiveFailures, ME.message);

            if consecutiveFailures >= cfg.ResetAfterConsecutiveFailures
                receiverResets = receiverResets + 1;
                warning('[Phase2] Receiver reset %d', receiverResets);
                localReleaseReceiver(rx);
                pause(cfg.PauseAfterReset);
                rx = localCreateReceiver(cfg);
                consecutiveFailures = 0;
                tail = complex(zeros(0,1,'single'));
            end
            drawnow limitrate;
            continue;
        end

        framesThisAttempt = framesThisAttempt + 1;
        numFramesTotal = numFramesTotal + 1;
        totalOverrun = totalOverrun + overrun;

        if len <= 0 || isempty(rxSig)
            drawnow limitrate;
            continue;
        end

        rxSig = single(rxSig(:));
        rxSig = rxSig(1:len);

        block = [tail; rxSig];
        block = block - mean(block);

        nTail = numel(tail);
        nBlock = numel(block);
        blockAbsStart = int64(totalSamplesRead) - int64(nTail) + 1;

        if nBlock < L
            tail = block;
            totalSamplesRead = totalSamplesRead + uint64(len);
            continue;
        end

        coarseStarts = localPacketDetectAll(block, cfg.WiFiChannelBandwidth, cfg.PacketDetectThreshold, 160);

        validMask = true(nBlock,1);
        validMask(1:min(nTail,nBlock)) = false;

        for k = 1:numel(coarseStarts)
            st = max(1, coarseStarts(k) - 240);
            ed = min(nBlock, coarseStarts(k) + 800);
            validMask(st:ed) = false;
        end

        candidateStarts = localFindValidStarts(validMask, L);
        candidateStarts = candidateStarts(candidateStarts > nTail);

        if ~isempty(candidateStarts)
            powers = inf(numel(candidateStarts),1);
            metrics = inf(numel(candidateStarts),1);

            for ii = 1:numel(candidateStarts)
                st = candidateStarts(ii);
                clip = block(st:st+L-1);
                powers(ii) = mean(abs(clip).^2);
                metrics(ii) = localSTFMetric(clip);
            end

            [~, ord] = sort(powers, 'ascend');
            candidateStarts = candidateStarts(ord);
            powers = powers(ord);
            metrics = metrics(ord);

            acceptedThisFrame = 0;
            for ii = 1:numel(candidateStarts)
                if numNeg >= Ln || acceptedThisFrame >= 4
                    break;
                end

                st = candidateStarts(ii);
                ed = st + L - 1;
                clip = block(st:ed);
                clipPow = powers(ii);
                stfMetric = metrics(ii);
                absStart = blockAbsStart + int64(st) - 1;

                if absStart <= lastAcceptedNegativeAbsStart + int64(cfg.NegativeMinSeparation)
                    continue;
                end
                if clipPow > cfg.MaxNegativePower
                    continue;
                end
                if stfMetric > cfg.MaxNegativeSTFMetric
                    continue;
                end

                localDetect0 = localPacketDetectFirst(clip, cfg.WiFiChannelBandwidth, cfg.PacketDetectThreshold);
                if ~isempty(localDetect0)
                    continue;
                end

                numNeg = numNeg + 1;
                acceptedThisFrame = acceptedThisFrame + 1;
                lastAcceptedNegativeAbsStart = absStart;

                xNeg(numNeg,:) = reshape(clip, 1, []);
                info.frameIndex(numNeg) = uint32(numFramesTotal);
                info.startIdx(numNeg)   = uint64(absStart);
                info.clipPower(numNeg)  = single(clipPow);
                info.stfMetric(numNeg)  = single(stfMetric);
                lastProgressFrame = numFramesTotal;

                if mod(numNeg, cfg.VerboseEvery) == 0 || numNeg == 1
                    fprintf('Negative %d / %d | frame=%d | absStart=%d | power=%.4e | STF=%.3f\n', ...
                        numNeg, Ln, numFramesTotal, absStart, clipPow, stfMetric);
                end
            end
        end

        keep = min(cfg.OverlapLength, nBlock);
        tail = block(end-keep+1:end);
        totalSamplesRead = totalSamplesRead + uint64(len);

        if mod(numFramesTotal, cfg.HeartbeatEveryFrames) == 0
            fprintf('Phase2 heartbeat | attempt=%d | frame=%d | neg=%d/%d | coarseDet=%d | overruns=%d | resets=%d | lastProgress=%d\n', ...
                attemptIdx, numFramesTotal, numNeg, Ln, numel(coarseStarts), totalOverrun, receiverResets, lastProgressFrame);
        end

        drawnow limitrate;
    end

    localReleaseReceiver(rx);

    if numNeg < Ln
        fprintf('[Phase2] Attempt %d ended without enough negatives. Restarting capture...\n', attemptIdx);
        pause(cfg.PauseAfterReset);
    end
end

xNeg = xNeg(1:numNeg,:);
info.frameIndex = double(info.frameIndex(1:numNeg));
info.startIdx   = double(info.startIdx(1:numNeg));
info.clipPower  = double(info.clipPower(1:numNeg));
info.stfMetric  = double(info.stfMetric(1:numNeg));

out = struct();
out.x = xNeg;
out.info = info;

end

%% ===================== Helpers =====================
function rx = localCreateReceiver(cfg)
rx = comm.SDRuReceiver( ...
    'Platform', 'N200/N210/USRP2', ...
    'IPAddress', cfg.RadioIP, ...
    'MasterClockRate', cfg.MasterClockRate, ...
    'DecimationFactor', cfg.DecimationFactor, ...
    'CenterFrequency', cfg.CenterFrequency, ...
    'Gain', cfg.Gain, ...
    'SamplesPerFrame', cfg.SamplesPerFrame, ...
    'OutputDataType', 'single');

if logical(cfg.UseBurstMode)
    rx.EnableBurstMode = true;
    rx.NumFramesInBurst = cfg.NumFramesInBurst;
end
end

function localReleaseReceiver(rx)
try
    if ~isempty(rx)
        release(rx);
    end
catch
end
end

function starts = localPacketDetectAll(x, cbw, threshold, minSpacing)
starts = [];
offset = 0;
N = size(x,1);

while offset < N-1
    try
        startOff = wlanPacketDetect(x, cbw, offset, threshold);
    catch
        startOff = [];
    end

    if isempty(startOff)
        break;
    end

    pktStart0 = offset + startOff; % zero-based
    pktStart1 = pktStart0 + 1;     % one-based

    if pktStart1 > N
        break;
    end

    starts(end+1,1) = pktStart1; %#ok<AGROW>
    offset = pktStart0 + max(1, minSpacing);
end

if ~isempty(starts)
    starts = unique(starts);
end
end

function startOff = localPacketDetectFirst(x, cbw, threshold)
try
    startOff = wlanPacketDetect(x, cbw, 0, threshold);
catch
    startOff = [];
end
end

function fineOff = localSymbolTiming(x, cbw, threshold)
try
    fineOff = wlanSymbolTimingEstimate(x, cbw, threshold);
catch
    try
        fineOff = wlanSymbolTimingEstimate(x, cbw);
    catch
        fineOff = [];
    end
end
if isempty(fineOff)
    fineOff = [];
end
end

function m = localSTFMetric(x)
L = min(numel(x), 160);
if L <= 16
    m = 0;
    return;
end
x = x(1:L);
num = abs(sum(x(1:end-16) .* conj(x(17:end))));
den = sum(abs(x(17:end)).^2) + eps;
m = num / den;
end

function starts = localFindValidStarts(validMask, clipLength)
validMask = validMask(:).';
convLen = conv(double(validMask), ones(1, clipLength), 'valid');
starts = find(convLen == clipLength).';
starts = starts(:);
end

function expectedFs = localExpectedSampleRate(wifiBw)
switch upper(wifiBw)
    case 'CBW20'
        expectedFs = 20e6;
    case 'CBW10'
        expectedFs = 10e6;
    case 'CBW5'
        expectedFs = 5e6;
    otherwise
        error('Unsupported bandwidth: %s', wifiBw);
end
end

function [outputFolder, outputStem] = localResolveOutputStem(outputFile)
[outputFolder, outputStem, ~] = fileparts(outputFile);
if isempty(outputStem)
    outputStem = 'wifi_preamble';
end
if isempty(outputFolder)
    outputFolder = pwd;
end
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
end

function localWriteCsvDataset(csvFile, x, y, chunkSize)
N = size(x,1);
L = size(x,2);

header = cell(1, 2*L + 1);
for k = 1:L
    header{2*k-1} = sprintf('I_%03d', k);
    header{2*k}   = sprintf('Q_%03d', k);
end
header{2*L + 1} = 'label';

fid = fopen(csvFile, 'w');
if fid < 0
    error('Failed to open CSV file for writing: %s', csvFile);
end
fprintf(fid, '%s\n', strjoin(header, ','));
fclose(fid);

for s = 1:chunkSize:N
    e = min(s + chunkSize - 1, N);
    xr = real(x(s:e,:));
    xq = imag(x(s:e,:));
    chunk = zeros(e - s + 1, 2*L + 1, 'single');
    chunk(:,1:2:2*L-1) = xr;
    chunk(:,2:2:2*L)   = xq;
    chunk(:,end)       = single(double(y(s:e)));
    writematrix(chunk, csvFile, 'WriteMode', 'append');
end
end