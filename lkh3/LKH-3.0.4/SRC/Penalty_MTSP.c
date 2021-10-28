#include "LKH.h"
#include "Segment.h"

GainType Penalty_MTSP_MINSUM()
{
    int Forward = SUCC(Depot)->Id != Depot->Id + DimensionSaved;
    Node *N = Depot, *NextN;
    GainType P = 0;
    int MinSize = INT_MAX, MaxSize = 0;

    do {
        int Size = -1;
        do {
            NextN = Forward ? SUCC(N) : PREDD(N);
            Size++;
            if (NextN->Id > DimensionSaved)
                NextN = Forward ? SUCC(NextN) : PREDD(NextN);
        } while ((N = NextN)->DepotId == 0);
        if (Size < MinSize)
            MinSize = Size;
        if (Size > MaxSize)
            MaxSize = Size;
    } while (N != Depot);

    if (MTSPMaxSize < Dimension - Salesmen && MaxSize > MTSPMaxSize)
        P += MaxSize - MTSPMaxSize;
    if (MTSPMinSize >= 1 && MinSize < MTSPMinSize)
        P += MTSPMinSize - MinSize;
    return P;
}

GainType Penalty_MTSP_MINMAX()
{
    int Forward = SUCC(Depot)->Id != Depot->Id + DimensionSaved;
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType Cost, MaxCost = MINUS_INFINITY;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > DimensionSaved)
        StartRoute -= DimensionSaved;
    N = StartRoute;
    do {
        Cost = 0;
        CurrentRoute = N;
        do {
            NextN = Forward ? SUCC(N) : PREDD(N);
            Cost += C(N, NextN) - N->Pi - NextN->Pi;
            if (NextN->Id > DimensionSaved)
                NextN = Forward ? SUCC(NextN) : PREDD(NextN);
        } while ((N = NextN)->DepotId == 0);
        Cost /= Precision;
        if (Cost > MaxCost) {
            if (Cost > CurrentPenalty ||
                (Cost == CurrentPenalty && CurrentGain <= 0)) {
                StartRoute = CurrentRoute;
                return CurrentPenalty + (CurrentGain > 0);
            }
            MaxCost = Cost;
        }
    } while (N != StartRoute);
    return MaxCost;
}

GainType Penalty_MTSP_MINMAX_SIZE()
{
    int Forward = SUCC(Depot)->Id != Depot->Id + DimensionSaved;
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    int Size, MaxSize = INT_MIN;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > DimensionSaved)
        StartRoute -= DimensionSaved;
    N = StartRoute;
    do {
        Size = 0;
        CurrentRoute = N;
        do {
            NextN = Forward ? SUCC(N) : PREDD(N);
            Size++;
            if (NextN->Id > DimensionSaved)
                NextN = Forward ? SUCC(NextN) : PREDD(NextN);
        } while ((N = NextN)->DepotId == 0);
        if (Size > MaxSize) {
            if (Size > CurrentPenalty ||
                (Size == CurrentPenalty && CurrentGain <= 0)) {
                StartRoute = CurrentRoute;
                return CurrentPenalty + (CurrentGain > 0);
            }
            MaxSize = Size;
        }
    } while (N != StartRoute);
    return MaxSize;
}
