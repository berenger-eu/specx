#include "Utils/SpMap.hpp"

#include "UTester.hpp"

class TestMap : public UTester< TestMap > {
    using Parent = UTester< TestMap >;

    void TestBasic(){
        {
            SpMap<int, int> map;

            UASSERTETRUE(map.getSize() == 0);
            UASSERTETRUE(map.exist(0) == false);
            UASSERTETRUE(map.exist(1) == false);

            map.add(0, 1);

            UASSERTETRUE(map.getSize() == 1);
            UASSERTETRUE(map.exist(0) == true);
            UASSERTETRUE(map.exist(1) == false);

            map.add(1, 2);

            UASSERTETRUE(map.getSize() == 2);
            UASSERTETRUE(map.exist(0) == true);
            UASSERTETRUE(map.exist(1) == true);

            UASSERTETRUE(map.find(0) == 1);
            UASSERTETRUE(map.find(1) == 2);
            UASSERTETRUE(bool(map.find(3)) == false);

            map.addOrReplace(1, 4);

            UASSERTETRUE(map.getSize() == 2);
            UASSERTETRUE(map.exist(0) == true);
            UASSERTETRUE(map.exist(1) == true);

            UASSERTETRUE(map.find(0) == 1);
            UASSERTETRUE(map.find(1) == 4);
            UASSERTETRUE(bool(map.find(3)) == false);

            UASSERTETRUE(map.remove(1) == true);

            UASSERTETRUE(map.getSize() == 1);
            UASSERTETRUE(map.exist(0) == true);
            UASSERTETRUE(map.exist(1) == false);

            UASSERTETRUE(map.find(0) == 1);
            UASSERTETRUE(bool(map.find(1)) == false);
            UASSERTETRUE(bool(map.find(3)) == false);

            UASSERTETRUE(map.remove(0) == true);

            UASSERTETRUE(map.getSize() == 0);
            UASSERTETRUE(map.exist(0) == false);
            UASSERTETRUE(map.exist(1) == false);

            UASSERTETRUE(bool(map.find(0)) == false);
            UASSERTETRUE(bool(map.find(1)) == false);
            UASSERTETRUE(bool(map.find(3)) == false);
        }
        {
            SpMap<void*, std::unique_ptr<int>> map;

            int v1 = 99;
            auto ptr = std::make_unique<int>(98);
            auto ptraddr = ptr.get();
            map.add(&v1, std::move(ptr));
            UASSERTETRUE(map.exist(&v1));

            auto res = map.find(&v1);
            UASSERTETRUE(bool(res));
            UASSERTETRUE(bool(res.value().get()));
            UASSERTETRUE(res.value().get().get() == ptraddr);

            UASSERTETRUE(bool(map.find(ptraddr)) == false);
        }
    }

    void SetTests() {
        Parent::AddTest(&TestMap::TestBasic, "Basic test for map");
    }
};

// You must do this
TestClass(TestMap)

