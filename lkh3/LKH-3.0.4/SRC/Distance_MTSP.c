#include "LKH.h"

/*
 * The Distance_MTSP function computes the transformed distance for
 * an mTSP instance.
 */

int Distance_MTSP(Node * Na, Node * Nb)
{
    const int M = INT_MAX / 2 / Precision;

    if (Fixed(Na, Nb))
        return 0;
    if (Forbidden(Na, Nb))
        return M;
    if (Na->DepotId != 0 && Nb->DepotId != 0)
        return 0;
    if (DimensionSaved != Dimension) {
        if (Na->DepotId != 0)
            Na = Na->Id <= DimensionSaved ? Depot :
                &NodeSet[Depot->Id + DimensionSaved];
        else if (Nb->DepotId != 0)
            Nb = Nb->Id <= DimensionSaved ? Depot :
                &NodeSet[Depot->Id + DimensionSaved];
    } else if (Dim != Dimension) {
        if (Na->Id > Dim)
            Na = Depot;
        if (Nb->Id > Dim)
            Nb = Depot;
    }
    return OldDistance(Na, Nb);
}
