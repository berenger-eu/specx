#include "Utils/SpModes.hpp"
#include "Utils/small_vector.hpp"

#include "UTester.hpp"

class TestArrayView : public UTester< TestArrayView > {
    using Parent = UTester< TestArrayView >;

    void CompareWithArray(const SpArrayView& view,
                          small_vector_base<long int> &shouldBe,
                          const int lineOffset){
        for(long int idx : view){
            UASSERTETRUE_OFFSET(idx < static_cast<long int>(shouldBe.size()), lineOffset);
            UASSERTEEQUAL_OFFSET(shouldBe[idx], 1L, lineOffset);
            shouldBe[idx] -= 1;
        }

        for(const long int& res : shouldBe){
            UASSERTEEQUAL_OFFSET(res, 0L, lineOffset);
        }
    }

    void TestBasic(){
        {
            SpArrayView view(10);

            {
                small_vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               1, 1, 1, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            {
                small_vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               0, 1, 1, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(7);
            {
                small_vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               0, 1, 0, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            view.removeItem(7);
            view.removeItem(10);
            {
                small_vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               0, 1, 0, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }
        }

        {
            for(int idxRemove = 0 ; idxRemove < 4 ; ++idxRemove){
                SpArrayView view(4);
                view.removeItem(idxRemove);
                small_vector<long int> shouldBe{1, 1, 1, 1};
                shouldBe[idxRemove] = 0;
                CompareWithArray(view, shouldBe, __LINE__);
            }
        }

        {
            SpArrayView view(0,10,3);

            {
                small_vector<long int> shouldBe{1, 0, 0,
                                               1, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            {
                small_vector<long int> shouldBe{1, 0, 0,
                                               1, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(3);
            {
                small_vector<long int> shouldBe{1, 0, 0,
                                               0, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            view.removeItem(7);
            view.removeItem(10);
            {
                small_vector<long int> shouldBe{1, 0, 0,
                                               0, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItems(6,9);
            {
                small_vector<long int> shouldBe{1};
                CompareWithArray(view, shouldBe, __LINE__);
            }
        }

        {
            SpArrayView view(0,10,3);

            {
                small_vector<long int> shouldBe{1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.addItem(12);
            UASSERTETRUE(view.getNbIntervals() == 1);
            {
                small_vector<long int> shouldBe{1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1};
                CompareWithArray(view, shouldBe, __LINE__);
            }


            view.addItem(16);
            UASSERTETRUE(view.getNbIntervals() == 2);
            {
                small_vector<long int> shouldBe{1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  0, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.addItem(17);
            UASSERTETRUE(view.getNbIntervals() == 2);
            {
                small_vector<long int> shouldBe{1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  0, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.addItem(2);
            UASSERTETRUE(view.getNbIntervals() == 3);
            {
                small_vector<long int> shouldBe{1, 0, 1,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  0, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }
        }
    }

    void SetTests() {
        Parent::AddTest(&TestArrayView::TestBasic, "Basic test for array view");
    }
};

// You must do this
TestClass(TestArrayView)

