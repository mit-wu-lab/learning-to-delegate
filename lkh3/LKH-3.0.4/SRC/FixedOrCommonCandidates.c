#include "LKH.h"

/* 
 * The FixedOrCommonCandidates function returns the number of fixed or
 * common candidate edges emanating from a given node, N.
 */

int FixedOrCommonCandidates(Node * N)
{
    int Count = 0;
    Candidate *NN;

    if (N->FixedTo2)
        return 2;
    if (!N->FixedTo1 && MergeTourFiles < 2)
        return 0;
    for (NN = N->CandidateSet; NN && NN->To; NN++)
        if (FixedOrCommon(N, NN->To))
            Count++;
    if (Count > 2)
        eprintf("Node %d has more than two required candidate edges",
                N->Id);
    return Count;
}
