classdef HARQEntity < handle
% HARQEntity  Chase-combining of complex RD frames across HARQ retransmissions.
%
% For N MRC-combined copies in independent AWGN the post-combining SNR
% improves by 10*log10(N) dB. Number of retransmissions is drawn from a
% Bernoulli chain with the configured PacketErrorRate.
%
% Usage:
%   harq = HARQEntity('PacketErrorRate', 0.10, 'MaxRetx', 3);
%   combined = harq.chaseCombine(rdComplex);

    properties
        MaxRetx         (1,1) double  = 3
        PacketErrorRate (1,1) double  = 0.15
        Verbose         (1,1) logical = false
    end

    properties (Access = private)
        Stream = []
    end

    methods
        function obj = HARQEntity(varargin)
            p = inputParser;
            p.addParameter('MaxRetx',         3);
            p.addParameter('PacketErrorRate', 0.15);
            p.addParameter('Verbose',         false);
            p.parse(varargin{:});
            obj.MaxRetx         = p.Results.MaxRetx;
            obj.PacketErrorRate = p.Results.PacketErrorRate;
            obj.Verbose         = p.Results.Verbose;
            obj.Stream          = RandStream('mt19937ar', 'Seed', 'shuffle');
        end

        function reset(~)
        end

        function combined = chaseCombine(obj, rdFrame)
            if isreal(rdFrame)
                combined = obj.combineMagnitude(rdFrame);
            else
                combined = obj.combineComplex(rdFrame);
            end
        end
    end

    methods (Access = private)
        function combined = combineComplex(obj, frame)
            sigPow = mean(abs(frame).^2, 'all');
            if sigPow <= 0; combined = abs(frame); return; end
            ntx = 1;
            while ntx <= obj.MaxRetx && rand(obj.Stream) < obj.PacketErrorRate
                ntx = ntx + 1;
            end
            acc = frame;
            for k = 2:ntx
                noise = sqrt(sigPow/2) * ...
                        (randn(obj.Stream, size(frame)) + 1j*randn(obj.Stream, size(frame)));
                acc = acc + (frame + noise);
            end
            if obj.Verbose && ntx > 1
                fprintf('[HARQ] combined %d copies\n', ntx);
            end
            combined = abs(acc / ntx);
        end

        function combined = combineMagnitude(obj, mag)
            sigPow = mean(mag(:).^2);
            if sigPow <= 0; combined = mag; return; end
            ntx = 1;
            while ntx <= obj.MaxRetx && rand(obj.Stream) < obj.PacketErrorRate
                ntx = ntx + 1;
            end
            acc = mag;
            for k = 2:ntx
                acc = acc + abs(mag + sqrt(sigPow/2) * ...
                    (randn(obj.Stream,size(mag)) + 1j*randn(obj.Stream,size(mag))));
            end
            combined = acc / ntx;
        end
    end
end