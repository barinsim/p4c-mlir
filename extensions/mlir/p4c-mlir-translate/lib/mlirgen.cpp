#include "mlirgen.h"

#include "frontends/p4/methodInstance.h"

namespace p4mlir {

namespace {

// Converts P4 location stored in 'loc' into its MLIR counterpart
mlir::Location loc(mlir::OpBuilder& builder, const IR::Node* node) {
    // TODO:
    CHECK_NULL(node);
    return mlir::FileLineColLoc::get(builder.getStringAttr("test/file.p4"), 42, 422);
}

// Turns MLIR type T into MLIR type `!p4.ref<T>`
mlir::Type wrappedIntoRef(mlir::OpBuilder& builder, mlir::Type type) {
    BUG_CHECK(!type.isa<RefType>(), "Ref type cannot be wrapped into another reference");
    return p4mlir::RefType::get(builder.getContext(), type);
}

// Return true if MLIR value 'value' is of type 'RefType'
bool isRef(mlir::Value value) { return value.getType().isa<p4mlir::RefType>(); }

// Returns true if p4 type 'type' is externally allocated.
// Externally allocated types are roughly the same types which standard refers to as "compile time
// allocated" types. The main aspect for the P4->MLIR translation is that they can be referenced
// only through p4.ref<>
bool isExternallyAllocatedType(const IR::Type* type) {
    return type->is<IR::Type_Extern>() || type->is<IR::Type_SpecializedCanonical>() ||
           type->is<IR::Type_Specialized>() || type->is<IR::Type_Control>() ||
           type->is<IR::Type_Parser>() || type->is<IR::P4Control>() || type->is<IR::P4Parser>() ||
           type->is<IR::Type_Table>();
}

bool isExternallyAllocatedType(mlir::Type type) {
    return type.isa<p4mlir::ExternClassType>() || type.isa<p4mlir::ControlType>() ||
           type.isa<p4mlir::ParserType>() || type.isa<p4mlir::TableType>();
}

// Creates block arguments for jump from 'bb' to 'succ'.
// The order of arguments corresponds to phi arguments stored within 'ssaInfo'.
// 'ssaInfo' should be also used to create block parameters to match the order
std::vector<mlir::Value> createBlockArgs(const SSAInfo& ssaInfo, const BasicBlock* bb,
                                         const BasicBlock* succ, const ValuesTable& valuesTable) {
    CHECK_NULL(bb, succ);
    std::vector<mlir::Value> rv;
    auto phiInfo = ssaInfo.getPhiInfo(succ);
    for (auto& [decl, phi] : phiInfo) {
        BUG_CHECK(phi.sources.count(bb), "Phi node does not contain argument for the block");
        auto id = phi.sources.at(bb).value();
        auto argVal = valuesTable.get(decl, id);
        rv.push_back(argVal);
    }
    return rv;
}

// Visitor which handles actual P4 type -> MLIR type conversion.
// This visitor is meant to be used from within TypeConvertor
class TypeConvertorVisitor : public Inspector
{
    mlir::OpBuilder& builder;
    P4::TypeMap* typeMap = nullptr;
    P4::ReferenceMap* refMap = nullptr;

    // Output of this pass
    mlir::Type output;

 public:
    TypeConvertorVisitor(mlir::OpBuilder& builder_, P4::TypeMap* typeMap_,
                         P4::ReferenceMap* refMap_)
        : builder(builder_), typeMap(typeMap_), refMap(refMap_) {
        CHECK_NULL(typeMap, refMap);
    }

    mlir::Type getOutput() const { BUG_CHECK(output, "Output not set"); return output; }

 private:
    profile_t init_apply(const IR::Node* node) override {
        BUG_CHECK(!output, "Output already set");
        return Inspector::init_apply(node);
    }

    void end_apply(const IR::Node*) override { BUG_CHECK(output, "Output not set"); }

    bool preorder(const IR::Node* node) override {
        if (node->is<IR::Type>()) {
            BUG_CHECK(false, "Unknown type");
        }
        BUG_CHECK(false, "Invalid node");
    }

    bool preorder(const IR::Type_InfInt*) override {
        // TODO: create special type
        auto type =  mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Signed);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Bits* bits) override {
        auto sign = bits->isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        int size = bits->size;
        auto type = mlir::IntegerType::get(builder.getContext(), size, sign);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Boolean*) override {
        auto type = mlir::IntegerType::get(builder.getContext(), 1, mlir::IntegerType::Signless);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Header* hdr) override {
        cstring name = hdr->name;
        auto type = p4mlir::HeaderType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        return setOutput(type);
    }

    bool preorder(const IR::Type_Struct* str) override {
        cstring name = str->name;
        auto type = p4mlir::StructType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        return setOutput(type);
    }

    bool preorder(const IR::Type_Control* control) override {
        cstring name = control->name;
        auto type = p4mlir::ControlType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        return setOutput(type);
    }

    bool preorder(const IR::Type_Parser* parser) override {
        cstring name = parser->name;
        auto type = p4mlir::ParserType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        return setOutput(type);
    }

    bool preorder(const IR::Type_MethodBase* method) override {
        std::vector<mlir::Type> paramTypes;
        std::vector<mlir::Type> retTypes;
        auto* params = method->parameters;
        std::transform(params->begin(), params->end(), std::back_inserter(paramTypes),
                       [&](auto* param) {
                           const IR::Type* p4type = param->type;
                           auto type = convert(p4type);
                           auto dir = param->direction;
                           // Out and InOut prameters are passed in as p4.ref
                           if (dir == IR::Direction::Out || dir == IR::Direction::InOut) {
                               type = wrappedIntoRef(builder, type);
                           }
                           // Externally allocated types are also passed in as p4.ref;
                           else if (isExternallyAllocatedType(type)) {
                               type = wrappedIntoRef(builder, type);
                           }
                           return type;
                       });
        if (method->returnType && !method->returnType->is<IR::Type_Void>()) {
            retTypes.push_back(convert(method->returnType));
        }
        auto type = builder.getFunctionType(paramTypes, retTypes);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Var* typeVar) override {
        llvm::StringRef name = typeVar->getVarName().c_str();
        auto type = p4mlir::TypeVarType::get(builder.getContext(), name);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Extern* ext) override {
        llvm::StringRef name = ext->getName().toString().c_str();
        auto type = p4mlir::ExternClassType::get(builder.getContext(), name, {});
        return setOutput(type);
    }

    bool preorder(const IR::Type_SpecializedCanonical* specialized) override {
        BUG_CHECK(specialized->substituted->is<IR::Type_Extern>(), "Expected extern class type");
        auto* ext = specialized->substituted->to<IR::Type_Extern>();
        llvm::StringRef name = ext->getName().toString().c_str();

        // Collect type arguments
        std::vector<mlir::Type> typeArgs;
        std::transform(specialized->arguments->begin(), specialized->arguments->end(),
                       std::back_inserter(typeArgs),
                       [&](const IR::Type* type) { return convert(type); });

        auto type = p4mlir::ExternClassType::get(builder.getContext(), name, typeArgs);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Table* table) override {
        cstring name = table->table->name;
        auto type = p4mlir::TableType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        return setOutput(type);
    }

    bool preorder(const IR::Type_BaseList* tuple) override {
        std::vector<mlir::Type> types;
        auto& components = tuple->components;
        std::transform(components.begin(), components.end(), std::back_inserter(types),
                       [&](const IR::Type* p4type) { return convert(p4type); });

        // If all inner types are p4mlir::DontcareType, we just return p4mlir::DontcareType instead
        // of the tuple. It simplifies recognition of default cases in parser transitions
        bool hasOnlyDontCare = std::all_of(types.begin(), types.end(), [](mlir::Type type) {
            return type.isa<p4mlir::DontcareType>();
        });
        if (hasOnlyDontCare) {
            return setOutput(p4mlir::DontcareType::get(builder.getContext()));
        }

        auto type = builder.getTupleType(types);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Dontcare*) override {
        auto type = p4mlir::DontcareType::get(builder.getContext());
        return setOutput(type);
    }

    bool preorder(const IR::Type_Unknown*) override {
        auto type = p4mlir::DontcareType::get(builder.getContext());
        return setOutput(type);
    }

    bool preorder(const IR::Type_Name* typeName) override {
        CHECK_NULL(typeName->path);
        auto* realType = refMap->getDeclaration(typeName->path, true)->to<IR::Type>();
        auto type = convert(realType);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Error* error) override {
        auto type = p4mlir::ErrorType::get(builder.getContext());
        return setOutput(type);
    }

    bool preorder(const IR::Type_Enum* enm) override {
        llvm::StringRef name = enm->getName().toString().c_str();
        auto type = p4mlir::EnumType::get(builder.getContext(), name);
        return setOutput(type);
    }

    bool preorder(const IR::Type_Set* set) override {
        auto elemType = convert(set->elementType);
        auto type = p4mlir::SetType::get(builder.getContext(), elemType);
        return setOutput(type);
    }

    // Sets output of this visitor.
    // Returns false for convenient use at the end of preorder method
    bool setOutput(mlir::Type type) {
        output = type;
        return false;
    }

    // Convenience method to resolve inner types of types.
    // Does not set output if this visitor
    mlir::Type convert(const IR::Type* type) const {
        TypeConvertorVisitor convertor(builder, typeMap, refMap);
        type->apply(convertor);
        return convertor.getOutput();
    }
};

}  // namespace

// Converts P4 type into corresponding MLIR type
mlir::Type TypeConvertor::toMLIRType(const IR::Type* p4type) const {
    CHECK_NULL(p4type);
    TypeConvertorVisitor convertor(builder, typeMap, refMap);
    p4type->apply(convertor);
    return convertor.getOutput();
}

void MLIRGenImplCFG::postorder(const IR::BoolLiteral* boolean) {
    auto type = toMLIRType(typeMap->getType(boolean));
    CHECK_NULL(type);
    mlir::Value val =
        builder.create<p4mlir::ConstantOp>(loc(builder, boolean), type, (int64_t)boolean->value);
    valuesTable.add(boolean, val);
}

void MLIRGenImplCFG::postorder(const IR::Constant* cst) {
    auto type = toMLIRType(typeMap->getType(cst));
    CHECK_NULL(type);
    BUG_CHECK(cst->fitsInt64(), "Not implemented");
    mlir::Value val = builder.create<p4mlir::ConstantOp>(loc(builder, cst), type, cst->asInt64());
    valuesTable.add(cst, val);
}

void MLIRGenImplCFG::postorder(const IR::ReturnStatement* ret) {
    builder.create<p4mlir::ReturnOp>(loc(builder, ret));
}

void MLIRGenImplCFG::postorder(const IR::AssignmentStatement* assign) {
    auto rValue = valuesTable.get(assign->right);

    const IR::IDeclaration* decl = nullptr;
    if (auto* pe = assign->left->to<IR::PathExpression>()) {
        CHECK_NULL(pe->path);
        decl = refMap->getDeclaration(pe->path, true);
    }

    // Write to an SSA register allocated variable
    if (decl && allocation.get(decl) == AllocType::REG) {
        mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, assign), rValue);
        ID ssaID = ssaInfo.getID(assign->left->to<IR::PathExpression>());
        valuesTable.add(decl, ssaID, value);
        return;
    }

    // Write to a stack allocated variable
    mlir::Value addr = valuesTable.getAddr(assign->left);
    builder.create<p4mlir::StoreOp>(loc(builder, assign), addr, rValue);
}

void MLIRGenImplCFG::postorder(const IR::Declaration_Variable* decl) {
    // Retrieves or creates init value
    auto createInitValue = [&](auto* decl) -> mlir::Value {
        if (!decl->initializer) {
            auto type = toMLIRType(typeMap->getType(decl));
            auto init = builder.create<p4mlir::UninitializedOp>(loc(builder, decl), type);
            return init;
        }
        return valuesTable.get(decl->initializer);
    };

    auto init = createInitValue(decl);

    // No need to allocate stack space for reg variables
    if (allocation.get(decl) == AllocType::REG) {
        auto value = init;
        if (decl->initializer) {
            auto type = toMLIRType(typeMap->getType(decl));
            value = builder.create<p4mlir::CopyOp>(loc(builder, decl), type, init);
        }
        ID ssaID = ssaInfo.getID(decl);
        valuesTable.add(decl, ssaID, value);
        return;
    }

    // Create space for stack allocated variables
    BUG_CHECK(allocation.get(decl) == AllocType::STACK, "Expected STACK allocation");
    auto type = toMLIRType(typeMap->getType(decl));
    auto refType = wrappedIntoRef(builder, type);
    mlir::Value addr = builder.create<p4mlir::AllocOp>(loc(builder, decl), refType);
    builder.create<p4mlir::StoreOp>(loc(builder, decl), addr, init);
    valuesTable.addAddr(decl, addr);
}

void MLIRGenImplCFG::postorder(const IR::Cast* cast) {
    CHECK_NULL(cast->destType);
    auto src = valuesTable.get(cast->expr);
    auto targetType = toMLIRType(cast->destType);
    mlir::Value value = builder.create<p4mlir::CastOp>(loc(builder, cast), targetType, src);
    valuesTable.add(cast, value);
}

void MLIRGenImplCFG::postorder(const IR::Operation_Relation* cmp) {
    auto lhsValue = valuesTable.get(cmp->left);
    auto rhsValue = valuesTable.get(cmp->right);

    auto selectKind = [](const IR::Operation_Relation* cmp) {
        if (cmp->is<IR::Equ>()) {
            return CompareOpKind::eq;
        } else if (cmp->is<IR::Neq>()) {
            return CompareOpKind::ne;
        } else if (cmp->is<IR::Lss>()) {
            return CompareOpKind::lt;
        } else if (cmp->is<IR::Leq>()) {
            return CompareOpKind::le;
        } else if (cmp->is<IR::Grt>()) {
            return CompareOpKind::gt;
        } else if (cmp->is<IR::Geq>()) {
            return CompareOpKind::ge;
        }
        BUG_CHECK(false, "Unknown comparison operator");
    };

    CompareOpKind kind = selectKind(cmp);

    auto res = builder.create<CompareOp>(loc(builder, cmp), kind, lhsValue, rhsValue);
    valuesTable.add(cmp, res);
}

void MLIRGenImplCFG::postorder(const IR::Member* mem) {
    // Callable reference does not generate any ops, skip it
    auto* p4type = typeMap->getType(mem, true);
    if (p4type->is<IR::Type_Method>()) {
        return;
    }

    // Reference of an error or enum constant generates p4.constant
    if (auto* typeExpr = mem->expr->to<IR::TypeNameExpression>()) {
        auto* typeName = typeExpr->typeName->to<IR::Type_Name>();
        CHECK_NULL(typeName, typeName->path);
        auto* typeDecl = refMap->getDeclaration(typeName->path, true)->to<IR::ISimpleNamespace>();
        auto* valueDecl = typeDecl->getDeclByName(mem->member)->to<IR::Declaration_ID>();
        CHECK_NULL(valueDecl);
        mlir::Value val = builder.create<p4mlir::ConstantOp>(
            loc(builder, mem), toMLIRType(typeExpr->typeName), symbols.getSymbol(valueDecl));
        valuesTable.add(mem, val);
        return;
    }

    mlir::Value baseValue = valuesTable.getUnchecked(mem->expr);
    auto type = toMLIRType(p4type);
    auto name = builder.getStringAttr(mem->member.toString().c_str());

    // Materialize member of a register allocated variable
    if (!isRef(baseValue)) {
        BUG_CHECK(isRead(),
                  "Member access to a register variable can be used only in a read context");
        mlir::Value val =
            builder.create<p4mlir::GetMemberOp>(loc(builder, mem), type, baseValue, name);
        valuesTable.add(mem, val);
        return;
    }

    BUG_CHECK(isRef(baseValue), "Expected STACK allocated base object");

    // Retrieve member reference of a stack allocated variable
    auto refType = wrappedIntoRef(builder, type);
    mlir::Value addr =
        builder.create<p4mlir::GetMemberRefOp>(loc(builder, mem), refType, baseValue, name);

    // Member variable is written, return just the reference
    if (isWrite()) {
        valuesTable.addAddr(mem, addr);
        return;
    }

    // Member variable is only read.
    // If this is the last member access within the path, materialize the value.
    // Otherwise return just a reference
    if (findContext<IR::Member>()) {
        valuesTable.addAddr(mem, addr);
        return;
    }
    mlir::Value val = builder.create<p4mlir::LoadOp>(loc(builder, mem), type, addr);
    valuesTable.add(mem, val);
}

void MLIRGenImplCFG::postorder(const IR::MethodCallExpression* call) {
    // 'MethodCallExpression' represents different types of calls, each of which
    // needs to generate different ops.
    // Figure out which call this is and generate correct mlir
    auto* instance = P4::MethodInstance::resolve(call, refMap, typeMap);
    auto loca = loc(builder, call);
    CHECK_NULL(instance);

    // Resolves call arguments to MLIR values, those consist of the additional arguments
    // (see 'CollectAdditionalParams' pass) and the actual P4 arguments
    auto getAdditionalOperands = [&]() {
        std::vector<const IR::Declaration_Variable*> additional;
        if (auto* actCall = instance->to<P4::ActionCall>()) {
            additional = additionalParams.get(actCall->action);
        } else if (auto* applyCall = instance->to<P4::ApplyMethod>()) {
            if (applyCall->isTableApply()) {
                auto* pe = call->method->to<IR::Member>()->expr->to<IR::PathExpression>();
                auto* table = refMap->getDeclaration(pe->path, true)->to<IR::P4Table>();
                CHECK_NULL(table);
                additional = additionalParams.get(table);
            }
        }
        std::vector<mlir::Value> rv;
        std::transform(additional.begin(), additional.end(), std::back_inserter(rv),
                       [&](const IR::Declaration_Variable* decl) {
                           BUG_CHECK(allocation.get(decl) == AllocType::STACK,
                                     "Expected STACK allocation");
                           return valuesTable.getAddr(decl);
                       });
        return rv;
    };

    auto getP4Operands = [&]() {
        std::vector<mlir::Value> rv;
        auto* args = call->arguments;
        auto* parameters = instance->originalMethodType->parameters;
        BUG_CHECK(args->size() <= parameters->size(), "More arguments than parameters");
        int size = args->size();
        for (int i = 0; i < size; ++i) {
            auto* param = parameters->getParameter(i);
            auto* arg = args->at(i);
            rv.push_back(valuesTable.getUnchecked(arg));
        }
        return rv;
    };

    // Collect additional + real MLIR operands
    std::vector<mlir::Value> operands;
    auto add = getAdditionalOperands();
    operands.insert(operands.end(), add.begin(), add.end());
    auto real = getP4Operands();
    operands.insert(operands.end(), real.begin(), real.end());

    // If there is less P4 arguments than P4 parameters, we fill the rest with
    // p4.control_plane_value values. This can happen only within a table definition for
    // directionless parameters
    auto* parameters = instance->originalMethodType->parameters;
    std::size_t missing = parameters->size() - real.size();
    for (std::size_t idx = 0; idx < missing; ++idx) {
        auto* param = parameters->getParameter(real.size() + idx);
        BUG_CHECK(param->direction == IR::Direction::None, "Expected no direction");
        auto type = toMLIRType(param->type);
        mlir::Value argVal =
            builder.create<p4mlir::ControlPlaneValueOp>(loc(builder, call), type, type);
        operands.push_back(argVal);
    }

    // Get type parameters of the call
    std::vector<mlir::Type> typeOperands;
    std::transform(call->typeArguments->begin(), call->typeArguments->end(),
                   std::back_inserter(typeOperands),
                   [&](const IR::Type* p4type) { return toMLIRType(p4type); });

    // Generate MLIR depending on type of the call
    if (auto* actCall = instance->to<P4::ActionCall>()) {
        auto name = symbols.getSymbol(actCall->action);
        builder.create<p4mlir::CallOp>(loca, name, operands);
    }
    else if (auto* builtin = instance->to<P4::BuiltInMethod>()) {
        BUG_CHECK(typeMap->getType(builtin->appliedTo, true)->is<IR::Type_Header>(),
                  "Not implemented");
        auto loca = loc(builder, call);
        cstring name = builtin->name;
        auto type = builder.getIntegerType(1);

        // Generates ops to retrieve the `__valid` member, either as a p4.ref<> or loaded value.
        // Returns the retrieved value
        auto getValidBit = [&](mlir::Value base) -> mlir::Value {
            auto fieldName = builder.getStringAttr("__valid");
            if (isRef(base)) {
                auto refType = wrappedIntoRef(builder, type);
                return builder.create<p4mlir::GetMemberRefOp>(loca, refType, base, fieldName);
            }
            return builder.create<p4mlir::GetMemberOp>(loca, type, base, fieldName);
        };

        auto member = getValidBit(valuesTable.getUnchecked(builtin->appliedTo));

        // Translate builtins of the header type into explicit operations on the __valid field
        if (name == "setValid" || name == "setInvalid") {
            auto cst = builder.create<p4mlir::ConstantOp>(loca, type, name == "setValid");
            builder.create<p4mlir::StoreOp>(loca, member, cst);
        } else if (name == "isValid") {
            mlir::Value value = member;
            if (isRef(value)) {
                value = builder.create<p4mlir::LoadOp>(loca, type, member);
            }
            valuesTable.add(call, value);
        } else {
            BUG_CHECK(false, "Unsupported builtin");
        }
    }
    else if (auto* externFunc = instance->to<P4::ExternFunction>()) {
        auto name = symbols.getSymbol(externFunc->method);

        // If relevant, retrieve return value type
        auto* p4RetType = externFunc->actualMethodType->returnType;
        std::vector<mlir::Type> retTypes;
        if (p4RetType && !p4RetType->is<IR::Type_Void>()) {
            retTypes.push_back(toMLIRType(p4RetType));
        }

        auto callOp = builder.create<p4mlir::CallOp>(loca, retTypes, name, typeOperands, operands);
        if (!callOp->getResults().empty()) {
            valuesTable.addUnchecked(call, callOp.getResult(0));
        }
    }
    else if (auto* applyCall = instance->to<P4::ApplyMethod>()) {
        BUG_CHECK(call->method->is<IR::Member>(), "Unexpected indirect call");
        mlir::Value base = valuesTable.getAddr(call->method->to<IR::Member>()->expr);
        builder.create<p4mlir::CallApplyOp>(loca, base, operands);
    }
    else if (auto* externMethod = instance->to<P4::ExternMethod>()) {
        auto name = symbols.getSymbol(externMethod->method);

        // If relevant, retrieve return value type
        auto* p4RetType = externMethod->actualMethodType->returnType;
        std::vector<mlir::Type> retTypes;
        if (p4RetType && !p4RetType->is<IR::Type_Void>()) {
            retTypes.push_back(toMLIRType(p4RetType));
        }

        // Retrieve value of the target object
        auto* methodRef = call->method->to<IR::Member>();
        CHECK_NULL(methodRef);
        mlir::Value target = valuesTable.getAddr(methodRef->expr);

        auto callOp = builder.create<p4mlir::CallMethodOp>(loca, retTypes, target, name,
                                                           typeOperands, operands);
        if (!callOp->getResults().empty()) {
            valuesTable.addUnchecked(call, callOp.getResult(0));
        }
    }
    else {
        BUG_CHECK(false, "Unsupported call type");
    }

    // P4 has copy-in/copy-out semantics for calls.
    // At this point written stack allocated variables must be copied back into
    // its original memory. This does not apply for the additional parameters
    auto* args = call->arguments;
    std::for_each(args->begin(), args->end(), [&](const IR::Argument* arg) {
        // Variables that were not passed as p4.ref<T> are not copied back
        mlir::Value argVal = valuesTable.getUnchecked(arg);
        if (!isRef(argVal)) {
            return;
        }
        // Variables of externally allocated types are also not copied back
        auto* p4type = typeMap->getType(arg->expression, true);
        if (isExternallyAllocatedType(p4type)) {
            return;
        }
        mlir::Value tmpAddr = valuesTable.getAddr(arg);
        auto type = tmpAddr.getType().cast<p4mlir::RefType>().getType();
        auto tmpVal = builder.create<p4mlir::LoadOp>(loc(builder, arg), type, tmpAddr);
        auto addr = valuesTable.getAddr(arg->expression);
        builder.create<p4mlir::StoreOp>(loc(builder, arg), addr, tmpVal);
    });
}

void MLIRGenImplCFG::postorder(const IR::Argument* arg) {
    // P4 has copy-in/copy-out semantics for calls.
    // Stack allocated variables must be copied into temporaries before a call,
    // and copied back after. The 'copy back' part is done while visiting
    // `MethodCallExpression`
    mlir::Value exprValue = valuesTable.getUnchecked(arg->expression);

    // A register allocated value is immutable, no need to copy them into temporaries.
    // Stack allocated variables passed as read-only arguments are already materialized
    // at this point
    if (!isRef(exprValue)) {
        valuesTable.add(arg, exprValue);
        return;
    }

    // Externally allocated types cannot be copied therefore do not follow copy-in/copy-out
    // semantics
    auto* p4type = typeMap->getType(arg->expression, true);
    if (isExternallyAllocatedType(p4type)) {
        return;
    }

    // Stack allocated variable passed as a writeable argument must be copied into
    // a temporary stack space first
    auto refType = exprValue.getType();
    auto type = refType.cast<p4mlir::RefType>().getType();
    mlir::Value tmpAddr = builder.create<p4mlir::AllocOp>(loc(builder, arg), refType);

    // Gets the direction of the parameter to which this argument is bound to
    auto getArgDir = [&]() -> std::optional<IR::Direction> {
        auto* ctxt = getContext();
        if (!ctxt) {
            return std::nullopt;
        }
        auto* call = ctxt->parent->node->to<IR::MethodCallExpression>();
        if (!call) {
            return std::nullopt;
        }
        auto* type = call->method->type->to<IR::Type_MethodBase>();
        if (!type) {
            return std::nullopt;
        }
        auto* param = type->parameters->getParameter(ctxt->child_index);
        if (!param) {
            return std::nullopt;
        }
        return param->direction;
    };

    // Initialize the allocated stack space.
    // Only inout stack arguments must be initialized
    auto dir = getArgDir();
    BUG_CHECK(dir.has_value(), "Could not retrieve argument direction");
    BUG_CHECK(dir == IR::Direction::InOut || dir == IR::Direction::Out,
              "At this point, read-only arguments must be already handled");
    if (dir == IR::Direction::InOut) {
        mlir::Value tmpVal = builder.create<p4mlir::LoadOp>(loc(builder, arg), type, exprValue);
        builder.create<p4mlir::StoreOp>(loc(builder, arg), tmpAddr, tmpVal);
    }

    valuesTable.addAddr(arg, tmpAddr);
}

void MLIRGenImplCFG::postorder(const IR::PathExpression* pe) {
    auto* type = typeMap->getType(pe, true);

    // Callable/table/match-kind references do not generate any ops and can be skipped
    if (type->is<IR::Type_MethodBase>() || type->is<IR::Type_MatchKind>()) {
        return;
    }

    // Unconditional parser transition does not have special AST node, it's just 'PathExpression'
    // placed as the last expression of the parser state CFG. We handle it separately
    if (type->is<IR::Type_State>()) {
        buildTransitionOp(pe);
        return;
    }

    // Retrieve declaration
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path, true);

    // References of SSA values do not generate any operations
    if (allocation.get(decl) == AllocType::REG) {
        // If the SSA value is read, associate it with this 'PathExpression'.
        // It simplifies retrieving the value later
        if (isRead()) {
            ID ssaID = ssaInfo.getID(pe);
            valuesTable.add(pe, valuesTable.get(decl, ssaID));
        }
        return;
    }

    // Load address of a compile-time allocated extern member (e.g. parser, control, table)
    if (allocation.get(decl) == AllocType::EXTERN_MEMBER) {
        CHECK_NULL(pe->path);
        auto name = builder.getStringAttr(pe->path->toString().c_str());
        mlir::Value baseValue = getSelfValue().value();
        auto mlirType = toMLIRType(type);
        auto refType = wrappedIntoRef(builder, mlirType);
        auto addr =
            builder.create<p4mlir::GetMemberRefOp>(loc(builder, pe), refType, baseValue, name);
        valuesTable.addAddr(pe, addr);
        return;
    }

    // Load address and materialize a compile-time constant member
    if (allocation.get(decl) == AllocType::CONSTANT_MEMBER) {
        BUG_CHECK(isRead(), "Expected read context");
        CHECK_NULL(pe->path);
        auto name = builder.getStringAttr(pe->path->toString().c_str());
        mlir::Value baseValue = getSelfValue().value();
        auto mlirType = toMLIRType(type);
        auto refType = wrappedIntoRef(builder, mlirType);
        auto addr =
            builder.create<p4mlir::GetMemberRefOp>(loc(builder, pe), refType, baseValue, name);
        // Do not materialize only to read member
        mlir::Value value = addr;
        if (!findContext<IR::Member>()) {
            value = builder.create<p4mlir::LoadOp>(loc(builder, pe), mlirType, addr);
        }
        valuesTable.addUnchecked(pe, value);
        return;
    }

    // EXTERN allocation is exclusively for externally allocated objects declared as parameters.
    // These are not materialized no matter the context
    if (allocation.get(decl) == AllocType::EXTERN) {
        mlir::Value addr = valuesTable.getAddr(decl);
        valuesTable.addAddr(pe, addr);
        return;
    }

    // Retrieve the address of the stack variable
    BUG_CHECK(allocation.get(decl) == AllocType::STACK, "Expected STACK allocation");
    mlir::Value addr = valuesTable.getAddr(decl);

    // Do not materialize the value if it will be written
    if (isWrite()) {
        valuesTable.addAddr(pe, addr);
        return;
    }

    // Even if only read, do not materialize a composite value only because of a member access
    BUG_CHECK(!findContext<IR::ArrayIndex>(), "Not implemented");
    if (findContext<IR::Member>()) {
        valuesTable.addAddr(pe, addr);
        return;
    }

    // Materialize the value in a read context
    BUG_CHECK(isRead(), "Value must be within a read context");
    auto refType = addr.getType().cast<p4mlir::RefType>();
    auto val = builder.create<p4mlir::LoadOp>(loc(builder, pe), refType.getType(), addr);
    valuesTable.add(pe, val);
}

void MLIRGenImplCFG::postorder(const IR::StructExpression* structExpr) {
    auto* p4type = typeMap->getType(structExpr, true)->to<IR::Type_StructLike>();
    CHECK_NULL(p4type);
    auto type = toMLIRType(p4type);

    // Retrieve the MLIR component values. The initializers are represented as {fieldName,
    // expression} pairs, which must be ordered correctly
    std::vector<mlir::Value> values;
    auto& fields = p4type->fields;
    std::transform(fields.begin(), fields.end(), std::back_inserter(values),
                   [&](const IR::StructField* field) {
                       cstring fieldName = field->getName();
                       auto& components = structExpr->components;
                       const IR::Expression* initExpr = nullptr;
                       for (auto* namedExpr : components) {
                           if (namedExpr->getName() == fieldName) {
                               initExpr = namedExpr->expression;
                               break;
                           }
                       }
                       CHECK_NULL(initExpr);
                       return valuesTable.getUnchecked(initExpr);
                   });

    // P4 states that a header initialized by a list expression has its validity bit set to `true`.
    // MLIR represents this bit as an explicit '__valid' member as the last field, for which we must
    // add 'true' value as the last list component
    if (type.isa<p4mlir::HeaderType>()) {
        mlir::Value cst = builder.create<p4mlir::ConstantOp>(loc(builder, structExpr),
                                                             builder.getIntegerType(1), true);
        values.push_back(cst);
    }

    // Create TupleOp
    mlir::Value tupleVal = builder.create<p4mlir::TupleOp>(loc(builder, structExpr), type, values);
    valuesTable.add(structExpr, tupleVal);
}

void MLIRGenImplCFG::postorder(const IR::ListExpression* listExpr) {
    // Retrieve the generated list values
    std::vector<mlir::Value> values;
    auto& components = listExpr->components;
    std::transform(components.begin(), components.end(), std::back_inserter(values),
                   [&](const IR::Expression* expr) { return valuesTable.getUnchecked(expr); });

    // P4 standard defines a product operator, which takes multiple sets (or singleton sets) as
    // input and returns a result of a set product operation, returning set of tuples. The AST does
    // not have a dedicated product node, instead each ListExpression in the context of
    // SelectCase is the product operation. In P4 dialect we generate explicit product op
    if (findContext<IR::SelectCase>()) {
        mlir::Value setVal = builder.create<p4mlir::SetProductOp>(loc(builder, listExpr), values);
        valuesTable.add(listExpr, setVal);
        return;
    }

    // Otherwise create TupleOp
    auto type = toMLIRType(typeMap->getType(listExpr, true));
    mlir::Value tupleVal = builder.create<p4mlir::TupleOp>(loc(builder, listExpr), type, values);
    valuesTable.add(listExpr, tupleVal);
}

void MLIRGenImplCFG::postorder(const IR::DefaultExpression* de) {
    // TODO: not sure if dontcare value is always the correct op for this AST
    auto type = toMLIRType(typeMap->getType(de, true));
    mlir::Value value = builder.create<p4mlir::DontcareOp>(loc(builder, de), type, type);
    valuesTable.add(de, value);
}

void MLIRGenImplCFG::postorder(const IR::Add* add) { handleArithmeticOp<p4mlir::AddOp>(add); }

void MLIRGenImplCFG::postorder(const IR::Sub* sub) { handleArithmeticOp<p4mlir::SubOp>(sub); }

void MLIRGenImplCFG::postorder(const IR::Mul* mul) { handleArithmeticOp<p4mlir::MulOp>(mul); }

void MLIRGenImplCFG::postorder(const IR::Range* range) {
    handleArithmeticOp<p4mlir::RangeOp>(range);
}

void MLIRGenImplCFG::postorder(const IR::Mask* mask) {
    handleArithmeticOp<p4mlir::MaskOp>(mask);
}

bool MLIRGenImplCFG::preorder(const IR::LOr* lOr) {
    buildLogicalOp(lOr);
    return false;
}

bool MLIRGenImplCFG::preorder(const IR::LAnd* lAnd) {
    buildLogicalOp(lAnd);
    return false;
}

bool MLIRGenImplCFG::preorder(const IR::IfStatement* ifStmt) {
    CHECK_NULL(currBlock);
    visit(ifStmt->condition);
    mlir::Value cond = valuesTable.get(ifStmt->condition);
    mlir::Block* tBlock = getMLIRBlock(currBlock->getTrueSuccessor());
    mlir::Block* fBlock = getMLIRBlock(currBlock->getFalseSuccessor());
    auto tArgs = createBlockArgs(ssaInfo, currBlock, currBlock->getTrueSuccessor(), valuesTable);
    auto fArgs = createBlockArgs(ssaInfo, currBlock, currBlock->getFalseSuccessor(), valuesTable);
    auto l = loc(builder, ifStmt);
    builder.create<mlir::cf::CondBranchOp>(l, cond, tBlock, tArgs, fBlock, fArgs);
    return false;
}

bool MLIRGenImplCFG::preorder(const IR::SelectExpression* selectExpr) {
    buildTransitionOp(selectExpr);
    return false;
}

mlir::Block* MLIRGenImplCFG::getMLIRBlock(const BasicBlock* p4block) const {
    BUG_CHECK(blocksTable.count(p4block), "Could not retrieve corresponding MLIR block");
    return blocksTable.at(p4block);
}

template <typename OpType>
void MLIRGenImplCFG::handleArithmeticOp(const IR::Operation_Binary* arithOp) {
    mlir::Value lValue = valuesTable.get(arithOp->left);
    mlir::Value rValue = valuesTable.get(arithOp->right);

    auto val = builder.create<OpType>(loc(builder, arithOp), lValue, rValue);
    valuesTable.add(arithOp, val);
}

std::optional<mlir::Value> MLIRGenImplCFG::getSelfValue() const { return selfValue; }

mlir::Type MLIRGenImplCFG::toMLIRType(const IR::Type* p4type) const {
    TypeConvertor convertor(builder, typeMap, refMap);
    return convertor.toMLIRType(p4type);
}

mlir::Operation* MLIRGenImplCFG::buildTransitionOp(
    std::variant<const IR::SelectExpression*, const IR::PathExpression*> node) {
    // Unconditional transition is just 'PathExpression'
    if (std::holds_alternative<const IR::PathExpression*>(node)) {
        auto* pe = std::get<const IR::PathExpression*>(node);
        BUG_CHECK(pe && typeMap->getType(pe, true)->is<IR::Type_State>(),
                  "Expected parser state type");

        // Transition to 'accept' and 'reject' is handled by separate ops in P4 dialect
        if (pe->path->toString() == IR::ParserState::accept) {
            return builder.create<p4mlir::ParserAcceptOp>(loc(builder, pe));
        }
        if (pe->path->toString() == IR::ParserState::reject) {
            return builder.create<p4mlir::ParserRejectOp>(loc(builder, pe));
        }

        // Retrieve symbol reference of the state
        auto* stateDecl = refMap->getDeclaration(pe->path, true)->to<IR::ParserState>();
        CHECK_NULL(stateDecl);
        auto stateSymbol = symbols.getSymbol(stateDecl);

        // Retrieve state arguments, those were gathered earlier by 'CollectAdditionalParams'
        std::vector<mlir::Value> argVals;
        auto params = additionalParams.get(stateDecl);
        std::transform(params.begin(), params.end(), std::back_inserter(argVals),
                       [&](const IR::Declaration_Variable* decl) {
                           BUG_CHECK(allocation.get(decl) == AllocType::STACK,
                                     "Expected STACK allocation");
                           return valuesTable.getAddr(decl);
                       });

        // Build TransitionOp
        auto transOp = builder.create<p4mlir::TransitionOp>(loc(builder, pe), stateSymbol, argVals);
        return transOp;
    }

    BUG_CHECK(std::holds_alternative<const IR::SelectExpression*>(node), "Unexpected expression");
    auto* selectExpr = std::get<const IR::SelectExpression*>(node);

    // Generate selector
    visit(selectExpr->select);

    // Create SelectTransitionOp and insert 1 block
    mlir::Value selectorVal = valuesTable.getUnchecked(selectExpr->select);
    auto selectOp =
        builder.create<p4mlir::SelectTransitionOp>(loc(builder, selectExpr), selectorVal);
    auto& body = selectOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Generate the cases
    visit(selectExpr->selectCases);

    // If no SelectTransitionDefaultCaseOp was generated, we synthesize one
    bool hasDefaultCase = std::any_of(body.begin(), body.end(), [](const mlir::Operation& op) {
        return dyn_cast<p4mlir::SelectTransitionDefaultCaseOp>(op);
    });
    if (!hasDefaultCase) {
        buildImplicitDefaultTransition();
    }

    builder.setInsertionPointAfter(selectOp);
    return selectOp;
}

p4mlir::SelectTransitionDefaultCaseOp MLIRGenImplCFG::buildImplicitDefaultTransition() {
    // Create SelectTransitionCaseDefaultOp and insert 1 block
    auto caseOp = builder.create<p4mlir::SelectTransitionDefaultCaseOp>(builder.getUnknownLoc());
    auto& body = caseOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Generate the transition which sets error.NoMatch
    auto noMatchSymbol = mlir::SymbolRefAttr::get(builder.getStringAttr("NoMatch"));
    builder.create<p4mlir::ParserRejectOp>(builder.getUnknownLoc(), noMatchSymbol);

    builder.setInsertionPointAfter(caseOp);
    return caseOp;
}

Operation* MLIRGenImplCFG::buildLogicalOp(const IR::Operation_Binary* binOp) {
    // AST nodes for logical operators represent implicit control flow due to the short-circuiting
    // semantics. We must create the MLIR basic blocks manually:
    // expr1 || expr2
    //
    // ->
    //
    // $1 = expr1;
    // $2 = constant true
    // $3 = cmp ne ($1, $2)
    // cond_br $3 bb^1 : bb^2($2)
    //
    // bb^1:
    // $4 = expr2
    // br bb^2($4)
    //
    // bb^2($5 : bool):
    // $6 = copy($5)

    BUG_CHECK(binOp->is<IR::LAnd>() || binOp->is<IR::LOr>(), "Expected LAnd or LOr");

    auto loca = loc(builder, binOp);

    // Create true (bb^1) and false (bb^2) blocks by splitting the current one
    mlir::Block* tBlock = builder.getBlock()->splitBlock(builder.getBlock()->end());
    mlir::Block* fBlock = tBlock->splitBlock(tBlock->end());
    auto boolType = toMLIRType(IR::Type_Boolean::get());
    fBlock->addArgument(boolType, loc(builder, binOp));

    // Generate expr1
    visit(binOp->left);
    mlir::Value lhsVal = valuesTable.get(binOp->left);

    // Now we generate the $2 = constant (see above).
    // The only difference between generating && or || is this constant.
    // 'true' for ||
    // 'false' for &&
    bool cst = binOp->is<IR::LOr>();
    mlir::Value cstVal = builder.create<p4mlir::ConstantOp>(loca, boolType, (int64_t)cst);

    // Generate the comparison and jump
    mlir::Value flagVal =
        builder.create<p4mlir::CompareOp>(loca, CompareOpKind::ne, lhsVal, cstVal);
    mlir::ValueRange fArgs = {cstVal};
    mlir::ValueRange tArgs = {};
    builder.create<mlir::cf::CondBranchOp>(loca, flagVal, tBlock, tArgs, fBlock, fArgs);

    // Generate expr2
    builder.setInsertionPointToEnd(tBlock);
    visit(binOp->right);
    mlir::Value rhsVal = valuesTable.get(binOp->right);
    builder.create<mlir::cf::BranchOp>(loc(builder, binOp), fBlock, mlir::ValueRange{rhsVal});

    // The CopyOp is redundant but makes the MLIR nicer and the copy gets removed later
    builder.setInsertionPointToEnd(fBlock);
    auto val = builder.create<p4mlir::CopyOp>(loca, fBlock->getArgument(0));
    valuesTable.add(binOp, val);

    return val.getOperation();
}

bool MLIRGenImplCFG::preorder(const IR::SelectCase* c) {
    // Default case has no keys and a different op
    auto keysType = toMLIRType(typeMap->getType(c->keyset, true));
    if (keysType.isa<p4mlir::DontcareType>()) {
        auto caseOp = builder.create<p4mlir::SelectTransitionDefaultCaseOp>(loc(builder, c));
        auto& body = caseOp.getBody().emplaceBlock();
        builder.setInsertionPointToEnd(&body);
        visit(c->state);
        builder.setInsertionPointAfter(caseOp);
        return false;
    }

    // Otherwise create SelectTransitionCaseOp and insert 1 block
    auto caseOp = builder.create<p4mlir::SelectTransitionCaseOp>(loc(builder, c));
    auto& body = caseOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Create SelectTransitionKeysOp and insert 1 block
    auto keysOp = builder.create<p4mlir::SelectTransitionKeysOp>(loc(builder, c));
    auto& keysBody = keysOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&keysBody);

    // Generate ops for the keyset and pass the value into InitOp.
    // The keyset must always be SetType. If keyset generation did not create one we synthesize one.
    // e.g.:
    // $1 = p4.constant 3
    // ->
    // $1 = p4.constant 3
    // $2 = p4.range($1, $1)
    visit(c->keyset);
    auto keysVal = valuesTable.get(c->keyset);
    if (!keysVal.getType().isa<p4mlir::SetType>()) {
        keysVal = builder.create<p4mlir::RangeOp>(loc(builder, c->keyset), keysVal, keysVal);
    }
    builder.create<p4mlir::InitOp>(loc(builder, c->keyset), keysVal, keysType);

    // Generate the transition
    builder.setInsertionPointAfter(keysOp);
    visit(c->state);

    builder.setInsertionPointAfter(caseOp);
    return false;
}

mlir::Operation* MLIRGenImpl::buildTransitionOp(const IR::ParserState* state) {
    CHECK_NULL(state);

    // Transition to 'accept' and 'reject' is handled by separate ops in P4 dialect
    if (state->getName() == IR::ParserState::accept) {
        return builder.create<p4mlir::ParserAcceptOp>(builder.getUnknownLoc());
    }
    if (state->getName() == IR::ParserState::reject) {
        return builder.create<p4mlir::ParserRejectOp>(builder.getUnknownLoc());
    }

    // Retrieve symbol reference of the state
    auto stateSymbol = symbols.getSymbol(state);

    // Retrieve state arguments, those were gathered earlier by 'CollectAdditionalParams'
    std::vector<mlir::Value> argVals;
    auto params = additionalParams.get(state);
    std::transform(params.begin(), params.end(), std::back_inserter(argVals),
                   [&](const IR::Declaration_Variable* decl) {
                       BUG_CHECK(allocation.get(decl) == AllocType::STACK,
                                 "Expected STACK allocation");
                       return valuesTable.getAddr(decl);
                   });

    // Build TransitionOp
    return builder.create<p4mlir::TransitionOp>(builder.getUnknownLoc(), stateSymbol, argVals);
}

bool MLIRGenImpl::preorder(const IR::P4Parser* parser) {
    // Build MLIR for the entire parser block
    auto* op = buildControlOrParser(parser);
    CHECK_NULL(op);

    // Everything must be generated from within this function, do not traverse further
    return false;
}

bool MLIRGenImpl::preorder(const IR::P4Control* control) {
    // Build MLIR for the entire control block
    auto* op = buildControlOrParser(control);
    CHECK_NULL(op);

    // Everything must be generated from within this function, do not traverse further
    return false;
}

bool MLIRGenImpl::preorder(const IR::ParserState* state) {
    // 'accept' and 'reject' states were synthesised just to make the type checking work.
    // They are not actually needed in P4 dialect
    if (state->getName() == IR::ParserState::accept ||
        state->getName() == IR::ParserState::reject) {
        return false;
    }

    // Collect FunctionType of this state.
    // Contrary to P4, P4 dialect has state parameters.
    // Those are collected by 'CollectAdditionalParams' pass earlier
    auto params = additionalParams.get(state);
    std::vector<mlir::Type> paramsTypes;
    std::transform(params.begin(), params.end(), std::back_inserter(paramsTypes),
                   [&](const IR::Declaration_Variable* decl) {
                       BUG_CHECK(allocation.get(decl) == AllocType::STACK,
                                 "Additional params must be STACK allocated");
                       auto type = toMLIRType(typeMap->getType(decl, true));
                       return wrappedIntoRef(builder, type);
                   });

    // Create StateOp and insert 1 block
    llvm::StringRef name = state->getName().toString().c_str();
    auto funcType = builder.getFunctionType(paramsTypes, {});
    auto stateOp = builder.create<p4mlir::StateOp>(loc(builder, state), name, funcType);
    auto& block = stateOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);

    // Save the values table before we create any mapping specific for the state body
    ValuesTable savedTable = valuesTable;

    // Add additional parameters as MLIR block parameters and bind them with their P4 counterparts
    std::for_each(params.begin(), params.end(), [&](const IR::Declaration_Variable* decl) {
        auto type = toMLIRType(typeMap->getType(decl, true));
        type = wrappedIntoRef(builder, type);
        auto addr = block.addArgument(type, loc(builder, state));
        valuesTable.addAddr(decl, addr);
    });

    // Find enclosing parser block
    auto* context = findContext<IR::P4Parser>();

    // Generate state body
    genMLIRFromCFG(context, cfgInfo.get(state), stateOp.getBody());

    // Restore the old values table
    valuesTable = std::move(savedTable);

    builder.setInsertionPointAfter(stateOp);
    return false;
}

bool MLIRGenImpl::preorder(const IR::P4Action* action) {
    // Create FunctionType type of this action.
    // The input types consist of the additional parameters and the real P4 parameters
    std::vector<mlir::Type> inputTypes;
    auto additional = additionalParams.get(action);
    std::transform(additional.begin(), additional.end(), std::back_inserter(inputTypes),
                   [&](const IR::Declaration_Variable* decl) {
                       auto type = toMLIRType(typeMap->getType(decl, true));
                       return wrappedIntoRef(builder, type);
                   });
    auto funcType = toMLIRType(typeMap->getType(action, true)).cast<mlir::FunctionType>();
    std::copy(funcType.getInputs().begin(), funcType.getInputs().end(),
              std::back_inserter(inputTypes));
    funcType = builder.getFunctionType(inputTypes, {});

    // Create ActionOp with 1 block
    llvm::StringRef name(action->getName().toString());
    auto actOp = builder.create<p4mlir::ActionOp>(loc(builder, action), name, funcType);
    auto& block = actOp.getBody().emplaceBlock();
    auto saved = builder.saveInsertionPoint();

    // Save the values table before we create any mapping specific for the action body
    ValuesTable savedTable = valuesTable;

    // Add additional parameters as MLIR block parameters and bind them with their P4 counterparts
    std::for_each(additional.begin(), additional.end(), [&](const IR::Declaration_Variable* decl) {
        BUG_CHECK(allocation.get(decl) == AllocType::STACK, "Expected STACK allocation");
        auto type = toMLIRType(typeMap->getType(decl, true));
        auto refType = wrappedIntoRef(builder, type);
        auto addr = block.addArgument(refType, loc(builder, action));
        valuesTable.addAddr(decl, addr);
    });

    // Add P4 parameters as MLIR block parameters and bind them with their P4 counterparts
    auto* params = action->getParameters();
    std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
        auto type = toMLIRType(typeMap->getType(param, true));
        auto dir = param->direction;
        if (dir == IR::Direction::None || dir == IR::Direction::In) {
            BUG_CHECK(allocation.get(param) == AllocType::REG, "Expected REG allocation");
            auto arg = block.addArgument(type, loc(builder, action));
            ID ssaID = ssaInfo.getID(param);
            valuesTable.add(param, ssaID, arg);
        } else {
            BUG_CHECK(allocation.get(param) == AllocType::STACK, "Expected STACK allocation");
            type = wrappedIntoRef(builder, type);
            auto arg = block.addArgument(type, loc(builder, action));
            valuesTable.addAddr(param, arg);
        }
    });

    // Find enclosing control block (does not have to exist)
    auto* context = findContext<IR::P4Control>();

    // Generate action body
    genMLIRFromCFG(context, cfgInfo.get(action), actOp.getBody());

    // Restore the old values table
    valuesTable = std::move(savedTable);

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Method* method) {
    // P4 has overloaded extern methods/functions, which must be 'unoverloaded' for MLIR.
    // The name consists of the original P4 name + '_' + <number of parameters>
    std::size_t numParams = method->getParameters()->size();
    std::string newName = method->getName().toString() + "_" + std::to_string(numParams);

    // Collect type parameters
    std::vector<mlir::StringRef> typeParams;
    auto& p4TypeParams = method->type->typeParameters->parameters;
    std::transform(p4TypeParams.begin(), p4TypeParams.end(), std::back_inserter(typeParams),
                   [](const IR::Type_Var* typeVar) { return typeVar->getVarName().c_str(); });

    // Generate MLIR func type of this method/function
    auto funcType = toMLIRType(typeMap->getType(method, true)).cast<mlir::FunctionType>();

    // Check if this method is extern constructor, for which we generate different op
    auto* ext = findContext<IR::Type_Extern>();
    if (ext && ext->getName().toString() == method->getName().toString()) {
        BUG_CHECK(typeParams.empty(), "Unexpected type parameters for extern constructor");
        builder.create<p4mlir::ConstructorOp>(loc(builder, method), newName, funcType);
        return false;
    }

    // Create ExternOp
    auto actOp = builder.create<p4mlir::ExternOp>(loc(builder, method), newName, funcType,
                                                  builder.getStrArrayAttr(typeParams));
    return false;
}

void MLIRGenImpl::genMLIRFromCFG(P4Block context, CFG cfg, mlir::Region& targetRegion) {
    BUG_CHECK(targetRegion.hasOneBlock(), "Target region must have the entry block");
    auto saved = builder.saveInsertionPoint();

    // For each BasicBlock create MLIR Block and insert it into the 'targetRegion'
    CFGWalker::controlFlowTraversal(cfg.getEntry(), [&](BasicBlock* bb) {
        mlir::Block* block = nullptr;
        if (bb == cfg.getEntry()) {
            // MLIR block for the entry block is already created
            block = &targetRegion.front();
        } else {
            block = &targetRegion.emplaceBlock();
        }
        blocksTable.insert({bb, block});
    });

    // For each mlir::Block insert block arguments using phi nodes info.
    // These block arguments must be then bound to {decl, ssa id} pairs,
    // so that references can be resolved during MLIRgen
    CFGWalker::preorder(cfg.getEntry(), [&](BasicBlock* bb) {
        mlir::Block* block = blocksTable.at(bb);
        auto phiInfo = ssaInfo.getPhiInfo(bb);
        for (auto& [decl, phi] : phiInfo) {
            auto loc = builder.getUnknownLoc();
            auto type = toMLIRType(typeMap->getType(decl->to<IR::Declaration>(), true));
            BUG_CHECK(allocation.get(decl) == AllocType::REG,
                      "Unexpected allocation for a block argument");
            mlir::BlockArgument arg = block->addArgument(type, loc);
            // Bind the arg
            ID ssaID = phi.destination.value();
            valuesTable.add(decl, ssaID, arg);
        }
    });

    // Optionally create `self` value which is used to access member variables
    std::optional<mlir::Value> selfValue;
    if (context) {
        BUG_CHECK(blocksTable.count(cfg.getEntry()), "Could not retrieve entry MLIR block");
        auto* block = blocksTable.at(cfg.getEntry());
        builder.setInsertionPointToStart(block);
        const IR::Type* p4type = context.getType();
        auto contextType = toMLIRType(p4type);
        auto refType = wrappedIntoRef(builder, contextType);
        selfValue = builder.create<p4mlir::SelfOp>(loc(builder, context.toNode()), refType);
    }

    // Fill the MLIR Blocks with Ops
    auto* cfgMLIRGen = createCFGVisitor(selfValue);
    CFGWalker::preorder(cfg.getEntry(), [&](BasicBlock* bb) {
        // Retrieve corresponding MLIR block
        BUG_CHECK(blocksTable.count(bb), "Could not retrieve MLIR block");
        auto* block = blocksTable.at(bb);
        builder.setInsertionPointToEnd(block);

        // Generate ops. This is done using different visitor on per-statement basis
        auto& comps = bb->components;
        std::for_each(comps.begin(), comps.end(), [&](auto* node) { cfgMLIRGen->apply(node, bb); });

        // If the generated block is not terminated, generate correct terminator. We have to
        // retrieve the current block from the builder since some expressions (logical operators)
        // generate multiple blocks and the original block is no longer the last generated one
        block = builder.getBlock();
        if (!block->empty() && block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            return;
        }
        BUG_CHECK(bb->succs.size() <= 1, "Non-terminated block can have at most 1 successor");

        // If the BasicBlock has 1 successor, generate unconditional jump
        if (bb->succs.size() == 1) {
            auto* succ = bb->succs.front();
            BUG_CHECK(blocksTable.count(succ), "Could not retrive MLIR block");
            auto args = createBlockArgs(ssaInfo, bb, succ, valuesTable);
            builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), blocksTable.at(succ), args);
            return;
        }

        // Insert p4.return for actions
        if (context.isControl() || context.isEmpty()) {
            builder.create<p4mlir::ReturnOp>(builder.getUnknownLoc());
            return;
        }

        // For parser insert p4.parser_reject for states and transition to start for apply method
        BUG_CHECK(context.isParser(), "Expected parser context");
        bool inState = dyn_cast<p4mlir::StateOp>(targetRegion.getParentOp());
        if (inState) {
            builder.create<p4mlir::ParserRejectOp>(builder.getUnknownLoc());
        } else {
            buildTransitionOp(context.getStartState());
        }
    });

    builder.restoreInsertionPoint(saved);
}

MLIRGenImplCFG* MLIRGenImpl::createCFGVisitor(std::optional<mlir::Value> selfValue) {
    return new MLIRGenImplCFG(builder, valuesTable, allocation, blocksTable, typeMap, refMap,
                              ssaInfo, selfValue, symbols, additionalParams);
}

mlir::Operation* MLIRGenImpl::buildControlOrParser(P4Block block) {
    // Retrieve name and apply/ctr types
    std::string name(block.getName());
    auto applyFuncType = toMLIRType(block.getApplyMethodType()).cast<mlir::FunctionType>();
    auto ctrFuncType = toMLIRType(block.getConstructorType()).cast<mlir::FunctionType>();

    // Creates ControlOp/ParserOp with 1 block
    auto createControlOrParserOp = [&](P4Block block) -> mlir::Operation* {
        if (block.isControl()) {
            auto op = builder.create<p4mlir::ControlOp>(loc(builder, block.toNode()), name,
                                                        applyFuncType, ctrFuncType);
            op.getBody().emplaceBlock();
            return op;
        }
        auto op = builder.create<p4mlir::ParserOp>(loc(builder, block.toNode()), name,
                                                   applyFuncType, ctrFuncType);
        op.getBody().emplaceBlock();
        return op;
    };

    auto* blockOp = createControlOrParserOp(block);
    auto& body = blockOp->getRegion(0).front();
    builder.setInsertionPointToStart(&body);

    // Goes over iterators of the same length and ties them into a vector of pairs
    auto tieIntoPairs = [](auto paramIt, auto typeIt, std::size_t size) {
        std::vector<std::pair<const IR::Parameter*, mlir::Type>> rv;
        while (size--) {
            rv.push_back({*paramIt++, *typeIt++});
        }
        return rv;
    };

    // Create {IR::Parameter, mlir::Type} pairs for all block parameters. First apply params,
    // second constructor params. It simplifies further processing
    std::size_t size = block.getApplyParameters()->size();
    auto paramsWithTypes =
        tieIntoPairs(block.getApplyParameters()->begin(), applyFuncType.getInputs().begin(), size);
    size = block.getConstructorParameters()->size();
    auto tmp = tieIntoPairs(block.getConstructorParameters()->begin(),
                            ctrFuncType.getInputs().begin(), size);
    paramsWithTypes.insert(paramsWithTypes.end(), tmp.begin(), tmp.end());

    // Add P4 apply/ctr parameters as MLIR block parameters
    std::for_each(paramsWithTypes.begin(), paramsWithTypes.end(), [&](auto paramWithType) {
        auto [param, type] = paramWithType;
        auto arg = body.addArgument(type, loc(builder, param));
        if (allocation.get(param) == AllocType::REG) {
            ID ssaID = ssaInfo.getID(param);
            valuesTable.add(param, ssaID, arg);
        } else {
            valuesTable.addAddr(param, arg);
        }
    });

    // Generate everything apart from the apply method and out-of-apply local declarations
    auto decls = block.getBodyDeclarations();
    std::for_each(decls.begin(), decls.end(), [&](const IR::IDeclaration* decl) {
        if (!decl->is<IR::Declaration_Variable>()) {
            visit(decl->to<IR::Declaration>());
        }
    });

    // Create ApplyOp with 1 block
    auto applyOp = builder.create<p4mlir::ApplyOp>(loc(builder, block.toNode()));
    auto& applyBlock = applyOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&applyBlock);

    // Generate the block body (which includes out-of-apply local declarations and apply method for
    // P4Control)
    BUG_CHECK(cfgInfo.contains(block.toNode()), "Could not find CFG");
    genMLIRFromCFG(block, cfgInfo.get(block.toNode()), applyOp.getBody());

    builder.setInsertionPointAfter(blockOp);
    return blockOp;
}

TableActionOp MLIRGenImpl::buildTableActionOp(const IR::Expression* expr) {
    CHECK_NULL(expr);
    // In cases where the action is only specified by its name (without arguments with direction
    // specified), the inner expression is just 'PathExpression' and not 'MethodCallExpression'.
    // This case is handled in TableActionOp by attaching the symbol reference
    if (auto* pe = expr->to<IR::PathExpression>()) {
        CHECK_NULL(pe->path);
        auto* actionDecl = refMap->getDeclaration(pe->path, true)->to<IR::P4Action>();
        CHECK_NULL(actionDecl);
        auto symbol = symbols.getSymbol(actionDecl);
        return builder.create<p4mlir::TableActionOp>(loc(builder, expr), symbol);
    }

    // Otherwise create TableAction op and insert 1 block
    auto actOp = builder.create<p4mlir::TableActionOp>(loc(builder, expr));
    auto saved = builder.saveInsertionPoint();
    auto& body = actOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Generate p4.self (might be needed if member/global variables are used)
    auto* control = findContext<IR::P4Control>();
    CHECK_NULL(control, control->type);
    mlir::Value selfValue = generateSelfValue(loc(builder, expr), builder, control).value();

    // Generate the expression
    auto* cfgMLIRGen = createCFGVisitor(selfValue);
    cfgMLIRGen->apply(expr);

    builder.restoreInsertionPoint(saved);
    return actOp;
}

mlir::Type MLIRGenImpl::toMLIRType(const IR::Type* p4type) const {
    TypeConvertor convertor(builder, typeMap, refMap);
    return convertor.toMLIRType(p4type);
}

std::optional<mlir::Value> MLIRGenImpl::generateSelfValue(mlir::Location loc,
                                                          mlir::OpBuilder& builder,
                                                          P4Block context) {
    std::optional<mlir::Value> selfValue;
    if (context) {
        const IR::Type* p4type = context.getType();
        auto contextType = toMLIRType(p4type);
        auto refType = wrappedIntoRef(builder, contextType);
        selfValue = builder.create<p4mlir::SelfOp>(loc, refType);
    }
    return selfValue;
}

bool MLIRGenImpl::preorder(const IR::Type_Header* hdr) {
    // Create HeaderOp and insert 1 block
    cstring name = hdr->name;
    auto hdrOp = builder.create<p4mlir::HeaderOp>(loc(builder, hdr), llvm::StringRef(name.c_str()));
    auto saved = builder.saveInsertionPoint();
    auto& block = hdrOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);

    // Generate member declarations
    visit(hdr->fields);

    // Generate validity bit member declaration, which must be the last member
    builder.create<p4mlir::ValidBitDeclOp>(loc(builder, hdr), "__valid");

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Type_Struct* str) {
    // Create StructOp and insert 1 block
    cstring name = str->name;
    auto strOp = builder.create<p4mlir::StructOp>(loc(builder, str), llvm::StringRef(name.c_str()));
    auto saved = builder.saveInsertionPoint();
    auto& block = strOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);

    // Generate member declarations
    visit(str->fields);

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Type_Extern* ext) {
    // Collect type parameters
    std::vector<mlir::StringRef> typeParams;
    auto& p4TypeParams = ext->typeParameters->parameters;
    std::transform(p4TypeParams.begin(), p4TypeParams.end(), std::back_inserter(typeParams),
                   [](const IR::Type_Var* typeVar) { return typeVar->getVarName().c_str(); });

    // Create ExternClassOp and insert 1 block
    llvm::StringRef name = ext->name.toString().c_str();
    auto extClassOp = builder.create<p4mlir::ExternClassOp>(loc(builder, ext), name,
                                                            builder.getStrArrayAttr(typeParams));
    auto& body = extClassOp.getBody().emplaceBlock();
    auto saved = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&body);

    // Generate Methods
    visit(ext->methods);

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::StructField* field) {
    auto* p4type = field->type;
    auto type = toMLIRType(p4type);
    cstring name = field->name;
    builder.create<p4mlir::MemberDeclOp>(loc(builder, field), llvm::StringRef(name), type);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Declaration* decl) {
    BUG_CHECK(decl->is<IR::Declaration_Instance>() || decl->is<IR::Declaration_Constant>(),
              "Expected Declaration_Instance or Declaration_Constant");

    // Create MemberDeclOp with 1 entry block within its initializer region
    auto type = toMLIRType(typeMap->getType(decl, true));
    cstring name = decl->getName();
    auto memberOp =
        builder.create<p4mlir::MemberDeclOp>(loc(builder, decl), llvm::StringRef(name), type);
    auto& initBlock = memberOp.getInitializer().emplaceBlock();
    auto saved = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&initBlock);

    // Generate p4.self (might be needed if member/global const variables are used in constructor)
    auto* context = findContext<IR::IContainer>();
    std::optional<mlir::Value> selfValue = generateSelfValue(loc(builder, decl), builder, context);

    // Generate the contructor arguments into the initializer region
    auto* cfgMLIRGen = createCFGVisitor(selfValue);
    cfgMLIRGen->apply(decl);

    // Retrieve the MLIR values for the arguments
    std::vector<mlir::Value> argValues;
    if (auto* declCst = decl->to<IR::Declaration_Constant>()) {
        CHECK_NULL(declCst->initializer);
        argValues.push_back(valuesTable.get(declCst->initializer));
    } else if (auto* declInstance = decl->to<IR::Declaration_Instance>()) {
        auto* args = declInstance->arguments;
        std::for_each(args->begin(), args->end(),
                      [&](const IR::Argument* arg) { argValues.push_back(valuesTable.get(arg)); });
    }

    // Generate the p4.init terminator and pass the generated arguments into it.
    // If the declared type is extern class, we also add the name of the used constructor
    if (type.isa<p4mlir::ExternClassType>()) {
        auto* p4type = typeMap->getType(decl, true);
        if (auto* specialized = p4type->to<IR::Type_SpecializedCanonical>()) {
            p4type = specialized->substituted;
        }
        auto* ext = p4type->to<IR::Type_Extern>();
        auto* declInstance = decl->to<IR::Declaration_Instance>();
        CHECK_NULL(ext, declInstance);
        auto* ctrMethod = ext->lookupConstructor(declInstance->arguments);
        CHECK_NULL(ctrMethod);
        auto symbol = symbols.getSymbol(ctrMethod);
        builder.create<p4mlir::InitOp>(loc(builder, decl), symbol, argValues, type);
    } else {
        builder.create<p4mlir::InitOp>(loc(builder, decl), argValues, type);
    }

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::P4Table* table) {
    // Given declaration, returns its MLIR type wrapped into p4.ref
    auto getType = [&](const IR::Declaration_Variable* decl) {
        auto type = toMLIRType(typeMap->getType(decl, true));
        auto refType = wrappedIntoRef(builder, type);
        return refType;
    };

    // Given vector of declarations, returns their MLIR types wrapped into p4.ref
    auto getTypes = [&](const std::vector<const IR::Declaration_Variable*>& decls) {
        std::vector<mlir::Type> rv;
        std::transform(decls.begin(), decls.end(), std::back_inserter(rv), getType);
        return rv;
    };

    // Create TableOp and insert 1 block
    llvm::StringRef name = table->getName().toString().c_str();
    auto applyType = builder.getFunctionType({}, getTypes(additionalParams.get(table)));
    auto tableOp = builder.create<p4mlir::TableOp>(loc(builder, table), name, applyType);
    auto& body = tableOp.getBody().emplaceBlock();
    auto saved = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&body);

    // Save the values table before we create any mapping specific for the table body
    ValuesTable savedTable = valuesTable;

    // Create block arguments for the additional table parameters and create mapping between them
    std::vector<const IR::Declaration_Variable*> additional = additionalParams.get(table);
    std::for_each(additional.begin(), additional.end(), [&](const IR::Declaration_Variable* decl) {
        mlir::Value addr = body.addArgument(getType(decl), loc(builder, decl));
        valuesTable.addAddr(decl, addr);
    });

    // Generate table properties
    visit(table->properties->properties);

    // Restore the values table
    valuesTable = std::move(savedTable);

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Property* property) { return true; }

bool MLIRGenImpl::preorder(const IR::ExpressionValue* exprVal) {
    // Retrieve property name
    llvm::StringRef name = findContext<IR::Property>()->getName().toString().c_str();

    // 'default_action' property gets a special TableDefaultActionOp op in P4 dialect
    mlir::Block* body = nullptr;
    if (name == "default_action") {
        auto defOp = builder.create<p4mlir::TableDefaultActionOp>(loc(builder, exprVal));
        body = &defOp.getBody().emplaceBlock();
        builder.setInsertionPointToEnd(body);
        buildTableActionOp(exprVal->expression);
        builder.setInsertionPointAfter(defOp);
        return false;
    }

    // Create TablePropertyOp and insert 1 block
    auto propOp = builder.create<p4mlir::TablePropertyOp>(loc(builder, exprVal), name);
    body = &propOp.getBody().emplaceBlock();
    auto saved = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(body);

    // Generate p4.self (might be needed if global/member variables are used)
    auto* control = findContext<IR::P4Control>();
    CHECK_NULL(control);
    mlir::Value selfValue = generateSelfValue(loc(builder, exprVal), builder, control).value();

    // Generate the inner expression
    CHECK_NULL(exprVal->expression);
    auto* cfgMLIRGen = createCFGVisitor(selfValue);
    cfgMLIRGen->apply(exprVal->expression);

    // Pass the generated value into InitOp
    // TODO: if expression is MethodCallExpression which does return any value, no mapping
    // is created and this crashes, not sure what is the right solution here
    CHECK_NULL(exprVal->expression);
    auto type = toMLIRType(typeMap->getType(exprVal->expression, true));
    mlir::Value value = valuesTable.getUnchecked(exprVal->expression);
    builder.create<p4mlir::InitOp>(loc(builder, exprVal), value, type);

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::ActionList* actionsList) {
    // Create TableActionsList op and insert 1 block
    auto listOp = builder.create<p4mlir::TableActionsListOp>(loc(builder, actionsList));
    auto saved = builder.saveInsertionPoint();
    auto& body = listOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Generate the list elements
    auto& elems = actionsList->actionList;
    std::for_each(elems.begin(), elems.end(),
                  [&](const IR::ActionListElement* elem) { visit(elem); });

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::ActionListElement* actionElement) {
    buildTableActionOp(actionElement->expression);
    return false;
}

bool MLIRGenImpl::preorder(const IR::EntriesList* entriesList) {
    // Create TableEntriesList op and insert 1 block
    auto entriesOp = builder.create<p4mlir::TableEntriesListOp>(loc(builder, entriesList));
    auto saved = builder.saveInsertionPoint();
    auto& body = entriesOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Generate the list elements
    auto& entries = entriesList->entries;
    std::for_each(entries.begin(), entries.end(), [&](const IR::Entry* entry) { visit(entry); });

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Entry* entry) {
    // Create TableEntryOp op and insert 1 block
    auto entryOp = builder.create<p4mlir::TableEntryOp>(loc(builder, entry));
    auto saved = builder.saveInsertionPoint();
    auto& body = entryOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Create TableEntryKeysOp and insert 1 block
    auto keysOp = builder.create<p4mlir::TableEntryKeysOp>(loc(builder, entry));
    auto& keysBody = keysOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&keysBody);

    // Generate self-value (might be needed if const member variables are used within the entry
    // keys)
    auto* control = findContext<IR::P4Control>();
    CHECK_NULL(control);
    mlir::Value selfValue = generateSelfValue(loc(builder, entry), builder, control).value();

    // Generate MLIR for the ListExpression
    auto* cfgVisitor = createCFGVisitor(selfValue);
    cfgVisitor->apply(entry->keys);

    // Create InitOp and pass the generated ListExpression value into it
    auto listValue = valuesTable.get(entry->keys);
    std::vector<mlir::Value> value = {valuesTable.get(entry->keys)};
    auto type = toMLIRType(typeMap->getType(entry->keys, true));
    builder.create<p4mlir::InitOp>(loc(builder, entry->keys), value, type);

    // Generate entry action value
    builder.setInsertionPointAfter(keysOp);
    buildTableActionOp(entry->action);

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Key* keyList) {
    // Create TanbleKeysListOp and insert 1 block
    auto keysOp = builder.create<p4mlir::TableKeysListOp>(loc(builder, keyList));
    auto saved = builder.saveInsertionPoint();
    auto& body = keysOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Generate key elements
    visit(keyList->keyElements);

    builder.restoreInsertionPoint(saved);
    return true;
}

bool MLIRGenImpl::preorder(const IR::KeyElement* key) {
    // Retrieve matchKind symbol reference
    CHECK_NULL(key->matchType->path);
    auto* matchKindDecl =
        refMap->getDeclaration(key->matchType->path, true)->to<IR::Declaration_ID>();
    auto matchKindSymbol = symbols.getSymbol(matchKindDecl);

    // Create TanbleKeyOp and insert 1 block
    auto keysOp = builder.create<p4mlir::TableKeyOp>(loc(builder, key), matchKindSymbol);
    auto saved = builder.saveInsertionPoint();
    auto& body = keysOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Generate p4.self (might be needed if member/global variables are used)
    auto* control = findContext<IR::P4Control>();
    CHECK_NULL(control);
    mlir::Value selfValue = generateSelfValue(loc(builder, key), builder, control).value();

    // Generate the inner expression
    CHECK_NULL(key->expression);
    auto* cfgMLIRGen = createCFGVisitor(selfValue);
    cfgMLIRGen->apply(key);

    // Create InitOp and pass the generated value into it
    std::vector<mlir::Value> value = {valuesTable.getUnchecked(key->expression)};
    auto type = toMLIRType(typeMap->getType(key->expression, true));
    builder.create<p4mlir::InitOp>(loc(builder, key), value, type);

    builder.restoreInsertionPoint(saved);
    return true;
}

bool MLIRGenImpl::preorder(const IR::Declaration_MatchKind* matchKinds) {
    auto& elems = matchKinds->members;
    std::for_each(elems.begin(), elems.end(), [&](const IR::Declaration_ID* decl) {
        llvm::StringRef name = decl->getName().toString().c_str();
        builder.create<p4mlir::MatchKindOp>(loc(builder, decl), name);
    });
    return false;
}

bool MLIRGenImpl::preorder(const IR::Type_Error* error) {
    auto& elems = error->members;
    std::for_each(elems.begin(), elems.end(), [&](const IR::Declaration_ID* decl) {
        llvm::StringRef name = decl->getName().toString().c_str();
        builder.create<p4mlir::ErrorOp>(loc(builder, decl), name);
    });
    return false;
}

bool MLIRGenImpl::preorder(const IR::Type_Enum* enm) {
    // Create EnumOp and insert 1 block
    llvm::StringRef name = enm->getName().toString().c_str();
    auto enumOp = builder.create<p4mlir::EnumOp>(loc(builder, enm), name);
    auto& body = enumOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    // Create EnumeratorOp for each member
    auto& elems = enm->members;
    std::for_each(elems.begin(), elems.end(), [&](const IR::Declaration_ID* decl) {
        llvm::StringRef name = decl->getName().toString().c_str();
        builder.create<p4mlir::EnumeratorOp>(loc(builder, decl), name);
    });

    builder.setInsertionPointAfter(enumOp);
    return false;
}

bool MLIRGenImpl::preorder(const IR::ExpressionListValue*) { BUG_CHECK(false, "Not implemented"); }

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& context,
                                          const IR::P4Program* program) {
    mlir::OpBuilder builder(&context);
    auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(moduleOp.getBody());

    MLIRGen gen(builder);
    program->apply(gen);

    if (!program || ::errorCount() > 0) {
        return nullptr;
    }
    if (failed(mlir::verify(moduleOp))) {
        // Dump for debugging purposes
        moduleOp->print(llvm::outs());
        moduleOp.emitError("module verification error");
        return nullptr;
    }

    return moduleOp;
}

}  // namespace p4mlir
