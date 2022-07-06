///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPABSTRACTBUFFERMANAGER_HPP
#define SPABSTRACTBUFFERMANAGER_HPP

template <class TargetType>
class SpAbstractBufferManager{
public:
    virtual ~SpAbstractBufferManager(){}

    virtual TargetType* getABuffer() = 0;
    virtual void releaseABuffer(TargetType*) = 0;
};

#endif
