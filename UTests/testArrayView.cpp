#include "Utils/SpModes.hpp"

#include "UTester.hpp"

class TestArrayView : public UTester< TestArrayView > {
    using Parent = UTester< TestArrayView >;

    void CompareWithArray(const SpArrayView& view,
                          std::vector<long int> shouldBe,
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
                std::vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               1, 1, 1, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            {
                std::vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               0, 1, 1, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(7);
            {
                std::vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               0, 1, 0, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            view.removeItem(7);
            view.removeItem(10);
            {
                std::vector<long int> shouldBe{1, 1, 1, 1, 1,
                                               0, 1, 0, 1, 1};
                CompareWithArray(view, shouldBe, __LINE__);
            }
        }

        {
            for(int idxRemove = 0 ; idxRemove < 4 ; ++idxRemove){
                SpArrayView view(4);
                view.removeItem(idxRemove);
                std::vector<long int> shouldBe{1, 1, 1, 1};
                shouldBe[idxRemove] = 0;
                CompareWithArray(view, shouldBe, __LINE__);
            }
        }

        {
            SpArrayView view(0,10,3);

            {
                std::vector<long int> shouldBe{1, 0, 0,
                                               1, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            {
                std::vector<long int> shouldBe{1, 0, 0,
                                               1, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(3);
            {
                std::vector<long int> shouldBe{1, 0, 0,
                                               0, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItem(5);
            view.removeItem(7);
            view.removeItem(10);
            {
                std::vector<long int> shouldBe{1, 0, 0,
                                               0, 0, 0,
                                               1, 0, 0,
                                               1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.removeItems(6,9);
            {
                std::vector<long int> shouldBe{1};
                CompareWithArray(view, shouldBe, __LINE__);
            }
        }

        {
            SpArrayView view(0,10,3);

            {
                std::vector<long int> shouldBe{1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1};
                CompareWithArray(view, shouldBe, __LINE__);
            }

            view.addItem(12);
            UASSERTETRUE(view.getNbIntervals() == 1);
            {
                std::vector<long int> shouldBe{1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1, 0, 0,
                                  1};
                CompareWithArray(view, shouldBe, __LINE__);
            }


            view.addItem(16);
            UASSERTETRUE(view.getNbIntervals() == 2);
            {
                std::vector<long int> shouldBe{1, 0, 0,
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
                std::vector<long int> shouldBe{1, 0, 0,
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
                std::vector<long int> shouldBe{1, 0, 1,
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

