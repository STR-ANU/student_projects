function [col_tracker, hist, pt_tracker] = tracking_alg_v1(PXY)
%% Initial calculations and storage arrays
col_tracker=[];
[lnd, fr, depth] = size(PXY);
hist = cat(3, PXY(:,1:2,1),PXY(:,1:2,2),PXY(:,1:2,3));
%% Primary Loop
for k =2:(fr-1)
    history = [];
    %% M matrix 
    M = zeros(lnd,lnd);
    [row,cols] = size(M);
    
    for r =1:row %kth frame
        for c =1:cols %k+1th frame
            M(r,c) = delta([hist(:,k-1,1:2),hist(:,k,1:2),PXY(:,k+1,1:2)], lnd, 2, r, r,c); 
            %x =[hist(:,k-1,2),hist(:,k,2),PXY(:,k+1,2)]
        end
    end
    %% B matrix and selection 
    B = zeros(lnd, lnd);
    l = zeros(row,1);
    pt_tracker =[];
    for a =1:row
        B = zeros(lnd, lnd);
        for i =1:row
            [val, li] = min(M(i,:));
            B(i,li) = 0.01+nansum(M(i,:))-M(i,li) + nansum(M(:,li))-M(i,li); 
        end
        [val, ind] = max(B, [], 2);
        [val, prow] = max(val);
        pcol = ind(prow);
        %py = li;
        pt_tracker = [pt_tracker;  prow, pcol,  B(prow,pcol),M(prow,pcol), PXY(pcol,k+1,1),PXY(pcol,k+1,2), PXY(pcol,k+1,3)]; %prow-pcol
        history = [history; prow, pcol, PXY(pcol,k+1,1),PXY(pcol,k+1,2),PXY(pcol,k+1,3)];
        M(prow,:) = repelem(nan,lnd);
        M(:,pcol) = repelem(nan,lnd);
    end
    
    %% Value Saving 
    history = sortrows(history,1);
    hist = [hist, cat(3, history(:,3),history(:,4),history(:,5))];
    
    pt_tracker = sortrows(pt_tracker,2);
    col_tracker = cat(3, col_tracker, pt_tracker);
end


